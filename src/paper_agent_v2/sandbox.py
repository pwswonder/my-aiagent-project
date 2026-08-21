from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import httpx


class FailureCategory(StrEnum):
    SYNTAX = "syntax"
    IMPORT = "import"
    SHAPE = "shape"
    DTYPE_DEVICE = "dtype_device"
    OOM_TIMEOUT = "oom_timeout"
    SEMANTIC_CONTRACT = "semantic_contract"
    SANDBOX = "sandbox"


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    output: str = ""


@dataclass(slots=True)
class SandboxResult:
    status: str
    checks: list[CheckResult] = field(default_factory=list)
    failure_category: FailureCategory | None = None
    message: str | None = None
    return_code: int | None = None

    def as_json(self) -> dict[str, object]:
        return asdict(self)


VALIDATION_SCRIPT = r"""
import compileall
import json
import shutil
import sys
import traceback

sys.path.insert(0, "/workspace")

checks = []

def record(name, function):
    try:
        value = function()
        checks.append({"name": name, "passed": True, "output": str(value) if value is not None else ""})
        return value
    except Exception:
        checks.append({"name": name, "passed": False, "output": traceback.format_exc()})
        print(json.dumps({"checks": checks}))
        raise

def compile_source():
    compile_root = "/tmp/package"
    shutil.copytree(".", compile_root, dirs_exist_ok=True)
    if not compileall.compile_dir(compile_root, quiet=1):
        raise RuntimeError("compile failed")
record("compileall", compile_source)

def import_modules():
    model_module = __import__("model", fromlist=["PaperModel"])
    input_module = __import__("example_inputs", fromlist=["make_example_inputs"])
    return model_module, input_module
modules = record("import", import_modules)
model = record("instantiate", lambda: modules[0].PaperModel())
inputs = modules[1].make_example_inputs()
output = record("dummy_forward", lambda: model(**inputs))
tensors = output if isinstance(output, tuple) else (output,)
record("output_shape", lambda: [list(item.shape) for item in tensors])

def semantic_contract():
    with open("architecture.json", encoding="utf-8") as source:
        architecture = json.load(source)
    expected_outputs = architecture.get("outputs", [])
    if len(expected_outputs) != len(tensors):
        raise ValueError(f"expected {len(expected_outputs)} outputs, received {len(tensors)}")
    for expected, actual in zip(expected_outputs, tensors):
        expected_shape = expected.get("shape", [])
        if len(expected_shape) != actual.ndim:
            raise ValueError(f"output rank mismatch: expected {expected_shape}, received {list(actual.shape)}")
        for expected_dimension, actual_dimension in zip(expected_shape, actual.shape):
            if isinstance(expected_dimension, int) and expected_dimension != actual_dimension:
                raise ValueError(
                    f"output shape mismatch: expected {expected_shape}, received {list(actual.shape)}"
                )
    expected_parameters = architecture.get("parameter_count")
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameters is not None:
        tolerance = max(1, round(expected_parameters * 0.05))
        if abs(expected_parameters - actual_parameters) > tolerance:
            raise ValueError(
                f"parameter count mismatch: expected {expected_parameters}, received {actual_parameters}"
            )
    return {"parameter_count": actual_parameters}
record("semantic_contract", semantic_contract)

def backward():
    floating = [item for item in tensors if getattr(item, "is_floating_point", lambda: False)()]
    if not floating:
        raise TypeError("no floating point output is available for backward")
    sum(item.float().mean() for item in floating).backward()
record("backward", backward)

def optimizer_step():
    import torch
    parameters = [item for item in model.parameters() if item.requires_grad]
    if not parameters:
        raise RuntimeError("model exposes no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    optimizer.step()
record("optimizer_step", optimizer_step)
print(json.dumps({"checks": checks}))
"""


def _classify(output: str, timed_out: bool = False) -> FailureCategory:
    lowered = output.lower()
    if timed_out or "out of memory" in lowered or "killed" in lowered:
        return FailureCategory.OOM_TIMEOUT
    if "syntaxerror" in lowered or "indentationerror" in lowered:
        return FailureCategory.SYNTAX
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return FailureCategory.IMPORT
    if "shape" in lowered or "size mismatch" in lowered or "mat1 and mat2" in lowered:
        return FailureCategory.SHAPE
    if "dtype" in lowered or "device" in lowered:
        return FailureCategory.DTYPE_DEVICE
    return FailureCategory.SEMANTIC_CONTRACT


def _timeout_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


class DockerSandbox:
    def __init__(
        self,
        image: str = "ai-paper-agent-sandbox:latest",
        *,
        timeout_seconds: int = 120,
        memory: str = "2g",
        cpus: float = 1.0,
        pids: int = 128,
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.pids = pids

    def validate(self, package_path: Path) -> SandboxResult:
        root = package_path.resolve()
        if not (root / "model.py").is_file():
            return SandboxResult(status="failed", failure_category=FailureCategory.SANDBOX, message="model.py missing")
        return self._execute(f"type=bind,src={root},dst=/workspace,readonly")

    def validate_volume(self, relative_path: str, volume_name: str) -> SandboxResult:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or not volume_name.replace("-", "").isalnum():
            return SandboxResult(
                status="failed", failure_category=FailureCategory.SANDBOX, message="unsafe volume mount"
            )
        mount = f"type=volume,src={volume_name},dst=/workspace,readonly,volume-subpath={relative.as_posix()}"
        return self._execute(mount)

    def _execute(self, mount: str) -> SandboxResult:
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(self.pids),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            mount,
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "python",
            "-I",
            "-c",
            VALIDATION_SCRIPT,
        ]
        clean_environment = {"PATH": os.environ.get("PATH", "")}
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=clean_environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = f"{_timeout_output(exc.stdout)}\n{_timeout_output(exc.stderr)}"
            return SandboxResult(
                status="failed",
                failure_category=FailureCategory.OOM_TIMEOUT,
                message=output[-8_000:],
            )
        except OSError as exc:
            return SandboxResult(status="failed", failure_category=FailureCategory.SANDBOX, message=str(exc))

        output = f"{completed.stdout}\n{completed.stderr}".strip()
        checks: list[CheckResult] = []
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
                checks = [CheckResult(**item) for item in payload.get("checks", [])]
                break
            except (json.JSONDecodeError, TypeError):
                continue
        if completed.returncode == 0:
            return SandboxResult(status="passed", checks=checks, return_code=0)
        return SandboxResult(
            status="failed",
            checks=checks,
            failure_category=_classify(output),
            message=output[-8_000:],
            return_code=completed.returncode,
        )


class RemoteSandbox:
    def __init__(self, base_url: str, storage_root: Path, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.storage_root = storage_root.resolve()
        self.timeout_seconds = timeout_seconds

    def validate(self, package_path: Path) -> SandboxResult:
        resolved = package_path.resolve()
        if self.storage_root not in resolved.parents:
            return SandboxResult(
                status="failed",
                failure_category=FailureCategory.SANDBOX,
                message="package is outside shared storage",
            )
        relative = str(resolved.relative_to(self.storage_root))
        try:
            response = httpx.post(
                f"{self.base_url}/validate",
                json={"artifact_relative_path": relative},
                timeout=self.timeout_seconds + 10,
            )
            response.raise_for_status()
            payload = response.json()
            payload["checks"] = [CheckResult(**item) for item in payload.get("checks", [])]
            return SandboxResult(**payload)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return SandboxResult(status="failed", failure_category=FailureCategory.SANDBOX, message=str(exc))


def save_validation_result(package_path: Path, result: SandboxResult) -> None:
    (package_path / "validation.json").write_text(json.dumps(result.as_json(), indent=2), encoding="utf-8")
