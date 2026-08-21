from __future__ import annotations

import tempfile
from pathlib import Path

from paper_agent_v2.generation.package_writer import write_package
from paper_agent_v2.ir import EvidenceRef, EvidenceSource, ModelGraphSpec, NodeSpec, TensorSpec
from paper_agent_v2.sandbox import DockerSandbox


def main() -> None:
    spec = ModelGraphSpec(
        name="SandboxSmoke",
        task="regression",
        inputs=[TensorSpec(name="x", shape=[2, 4])],
        nodes=[
            NodeSpec(
                id="projection",
                op="Linear",
                inputs=["x"],
                output="output",
                params={"in_features": 4, "out_features": 2},
                evidence_ids=["e1"],
            )
        ],
        outputs=[TensorSpec(name="output", shape=[2, 2])],
        evidence=[
            EvidenceRef(
                id="e1",
                source_type=EvidenceSource.PDF,
                page=1,
                quote="Sandbox smoke-test evidence.",
            )
        ],
        parameter_count=10,
    ).approve()
    with tempfile.TemporaryDirectory(prefix="paper-agent-sandbox-") as directory:
        package = write_package(
            spec,
            Path(directory),
            document_sha256="0" * 64,
            spec_version=1,
            provider="static",
            model="smoke",
        )
        result = DockerSandbox().validate(package.path)
        print(result.status)
        for check in result.checks:
            print(f"{check.name}: {'passed' if check.passed else 'failed'}")
        if result.status != "passed":
            raise SystemExit(result.message or "sandbox validation failed")


if __name__ == "__main__":
    main()
