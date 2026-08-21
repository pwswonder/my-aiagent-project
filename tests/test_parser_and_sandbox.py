from pathlib import Path

import pytest

from paper_agent_v2.parser import PdfValidationError, validate_pdf
from paper_agent_v2.sandbox import DockerSandbox, FailureCategory


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
