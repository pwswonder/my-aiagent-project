from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

TERMINAL = {"completed", "failed", "cancelled"}


def wait_for(client: httpx.Client, path: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in TERMINAL:
            return payload
        time.sleep(2)
    raise TimeoutError(path)


def evidence_coverage(spec: dict[str, Any]) -> float:
    assumptions = {
        item["field"].split(".")[1] for item in spec.get("assumptions", []) if item["field"].startswith("nodes.")
    }
    nodes = spec.get("nodes", [])
    covered = sum(bool(node.get("evidence_ids")) or node["id"] in assumptions for node in nodes)
    return covered / len(nodes) if nodes else 1.0


def run_case(client: httpx.Client, case: dict[str, str], *, approve: bool, timeout: int) -> dict[str, Any]:
    pdf = Path(case["pdf"])
    if not pdf.is_file():
        return {**case, "status": "missing_pdf"}
    with pdf.open("rb") as source:
        response = client.post(
            "/api/v2/documents",
            files={"file": (pdf.name, source, "application/pdf")},
        )
    response.raise_for_status()
    accepted = response.json()
    run = wait_for(client, f"/api/v2/runs/{accepted['analysis_run_id']}", timeout)
    result: dict[str, Any] = {**case, "document_id": accepted["document_id"], "analysis": run}
    if run["status"] != "completed":
        return result
    spec_response = client.get(f"/api/v2/documents/{accepted['document_id']}/spec")
    spec_response.raise_for_status()
    spec_payload = spec_response.json()
    spec = spec_payload["spec"]
    result.update(
        {
            "spec_version": spec_payload["version"],
            "spec_status": spec_payload["status"],
            "node_evidence_coverage": evidence_coverage(spec),
            "blocking_unresolved": sum(item.get("blocking", True) for item in spec.get("unresolved", [])),
        }
    )
    if not approve or result["blocking_unresolved"]:
        return result
    approval = client.post(f"/api/v2/documents/{accepted['document_id']}/spec/{spec_payload['version']}/approve")
    approval.raise_for_status()
    generation = client.post(f"/api/v2/documents/{accepted['document_id']}/generations")
    generation.raise_for_status()
    generation_payload = generation.json()
    generation_run = wait_for(client, f"/api/v2/runs/{generation_payload['run_id']}", timeout)
    state = client.get(f"/api/v2/generations/{generation_payload['generation_id']}")
    state.raise_for_status()
    result.update({"generation_run": generation_run, "generation": state.json()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path("var/benchmark-v2.json"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.api, timeout=120) as client:
        results = [run_case(client, case, approve=args.approve, timeout=args.timeout) for case in cases]
    measured = [item for item in results if item.get("status") != "missing_pdf"]
    summary = {
        "cases": len(results),
        "measured": len(measured),
        "ir_schema_completed": sum("spec_version" in item for item in measured),
        "full_evidence_coverage": sum(item.get("node_evidence_coverage") == 1.0 for item in measured),
        "generation_passed": sum(item.get("generation", {}).get("status") == "completed" for item in measured),
        "needs_review_or_failed": sum(
            item.get("spec_status") == "needs_review"
            or item.get("generation", {}).get("status") in {"needs_review", "failed"}
            for item in measured
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
