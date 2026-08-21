from pathlib import Path

import pytest

from paper_agent_v2.parser import SECTION_RE, PdfValidationError, validate_pdf
from paper_agent_v2.sandbox import DockerSandbox, FailureCategory


def test_numbered_section_heading_with_trailing_dot_is_recognized() -> None:
    match = SECTION_RE.match("3.1. Architecture")
    assert match is not None
    assert match.group(1).lower() == "architecture"


def test_proposed_method_and_references_headings_are_recognized() -> None:
    method = SECTION_RE.match("3. Proposed Method")
    references = SECTION_RE.match("References")

    assert method is not None and method.group(1).lower() == "proposed method"
    assert references is not None and references.group(1).lower() == "references"


def test_non_pdf_and_oversized_uploads_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"not a pdf")
    with pytest.raises(PdfValidationError, match="signature"):
        validate_pdf(path, 100)
    path.write_bytes(b"%PDF-" + b"x" * 100)
    with pytest.raises(PdfValidationError, match="exceeds"):
        validate_pdf(path, 20)


def test_missing_model_is_a_structured_sandbox_failure(tmp_path: Path) -> None:
    result = DockerSandbox().validate(tmp_path)
    assert result.status == "failed"
    assert result.failure_category == FailureCategory.SANDBOX


def test_sandbox_volume_path_traversal_is_rejected() -> None:
    result = DockerSandbox().validate_volume("../../etc", "ai-paper-agent-storage")
    assert result.status == "failed"
    assert result.failure_category == FailureCategory.SANDBOX
