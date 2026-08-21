from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from paper_agent_v2.generation.renderer import render_model
from paper_agent_v2.ir import ModelGraphSpec

GENERATOR_VERSION = "2.0.0a1"
TEMPLATE_DIR = Path(__file__).with_name("templates")


@dataclass(frozen=True, slots=True)
class GeneratedPackage:
    path: Path
    provenance: dict[str, object]


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    if not slug:
        raise ValueError("artifact name contains no safe characters")
    return slug[:80]


def _shape_literal(shape: list[int | str]) -> str:
    symbolic_defaults = {"B": 2, "H": 224, "W": 224, "T": 32, "L": 32, "N": 32}
    values = [symbolic_defaults.get(value, 2) if isinstance(value, str) else value for value in shape]
    return repr(values)


def write_package(
    spec: ModelGraphSpec,
    destination_root: Path,
    *,
    document_sha256: str,
    spec_version: int,
    provider: str,
    model: str,
    custom_modules: dict[str, str] | None = None,
    repair_log: list[dict[str, object]] | None = None,
) -> GeneratedPackage:
    rendered = render_model(spec, custom_modules)
    if rendered.custom_operations:
        raise ValueError(f"custom modules required for: {', '.join(rendered.custom_operations)}")

    repair_log = repair_log or []
    fingerprint = json.dumps(repair_log, sort_keys=True, separators=(",", ":"))
    artifact_key = hashlib.sha256(
        f"{document_sha256}:{spec_version}:{provider}:{model}:{GENERATOR_VERSION}:{fingerprint}".encode()
    ).hexdigest()[:20]
    destination_root.mkdir(parents=True, exist_ok=True)
    package_path = (destination_root.resolve() / f"{_safe_slug(spec.name)}-{artifact_key}").resolve()
    if destination_root.resolve() not in package_path.parents:
        raise ValueError("artifact path escaped its storage root")
    package_path.mkdir(parents=True, exist_ok=False)
    (package_path / "tests").mkdir()

    environment = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined, autoescape=False)
    context = {"spec": spec, "input_shapes": [_shape_literal(x.shape) for x in spec.inputs]}
    files = {
        "model.py": rendered.source,
        "config.py": environment.get_template("config.py.j2").render(**context),
        "example_inputs.py": environment.get_template("example_inputs.py.j2").render(**context),
        "tests/test_model.py": environment.get_template("test_model.py.j2").render(**context),
        "README.md": environment.get_template("README.md.j2").render(**context),
    }
    for relative, content in files.items():
        (package_path / relative).write_text(content.rstrip() + "\n", encoding="utf-8")

    provenance: dict[str, object] = {
        "document_sha256": document_sha256,
        "spec_version": spec_version,
        "spec_schema_version": spec.schema_version,
        "provider": provider,
        "model": model,
        "generator_version": GENERATOR_VERSION,
        "artifact_key": artifact_key,
        "repair_log": repair_log,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (package_path / "architecture.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (package_path / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (package_path / "validation.json").write_text(
        json.dumps({"status": "pending", "checks": []}, indent=2), encoding="utf-8"
    )
    return GeneratedPackage(path=package_path, provenance=provenance)
