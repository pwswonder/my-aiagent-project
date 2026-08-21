from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(abstract|introduction|related work|method(?:ology)?|"
    r"model|architecture|approach|proposed method|experiments?|results?|discussion|limitations?|"
    r"conclusion|references|appendix)\b",
    re.IGNORECASE,
)
GITHUB_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
CAPTION_RE = re.compile(r"^(?:figure|fig\.|table)\s*\d+", re.IGNORECASE)
EQUATION_RE = re.compile(r"(?:=|∑|∏|softmax|argmax|\|\|).{0,120}")


@dataclass(slots=True)
class PaperChunk:
    id: str
    page: int
    section: str
    kind: str
    text: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class ParsedPaper:
    sha256: str
    title: str
    chunks: list[PaperChunk]
    github_urls: list[str] = field(default_factory=list)
    architecture_pages: list[int] = field(default_factory=list)


class PdfValidationError(ValueError):
    pass


def validate_pdf(path: Path, max_bytes: int) -> None:
    if not path.is_file():
        raise PdfValidationError("uploaded file does not exist")
    size = path.stat().st_size
    if size == 0:
        raise PdfValidationError("empty uploads are not allowed")
    if size > max_bytes:
        raise PdfValidationError(f"PDF exceeds the {max_bytes} byte limit")
    if path.read_bytes()[:5] != b"%PDF-":
        raise PdfValidationError("file signature is not PDF")


def _chunk_id(page: int, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"p{page:04d}-{index:04d}-{digest}"


def parse_pdf(path: Path, max_bytes: int = 50 * 1024 * 1024) -> ParsedPaper:
    validate_pdf(path, max_bytes)
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to parse PDFs") from exc

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    chunks: list[PaperChunk] = []
    github_urls: set[str] = set()
    architecture_pages: set[int] = set()
    current_section = "front_matter"
    title = path.stem

    with fitz.open(path) as document:
        if document.page_count == 0:
            raise PdfValidationError("PDF contains no pages")
        metadata_title = (document.metadata or {}).get("title")
        if metadata_title and metadata_title.strip():
            title = metadata_title.strip()

        for page_index, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)
            for block_index, block in enumerate(blocks):
                text = re.sub(r"\s+", " ", str(block[4])).strip()
                if len(text) < 2:
                    continue
                section_match = SECTION_RE.match(text)
                if section_match and len(text) < 120:
                    current_section = section_match.group(1).lower()
                    if current_section == "proposed method":
                        current_section = "method"
                caption = CAPTION_RE.match(text)
                kind = (
                    "table" if caption and text.lower().startswith("table") else "caption" if caption else "paragraph"
                )
                if EQUATION_RE.search(text) and len(text) < 500:
                    kind = "equation"
                lowered = f"{current_section} {text}".lower()
                if any(term in lowered for term in ("architecture", "model", "method", "network", "figure")):
                    architecture_pages.add(page_index)
                github_urls.update(url.rstrip(".,);]") for url in GITHUB_RE.findall(text))
                chunks.append(
                    PaperChunk(
                        id=_chunk_id(page_index, block_index, text),
                        page=page_index,
                        section=current_section,
                        kind=kind,
                        text=text,
                        bbox=(float(block[0]), float(block[1]), float(block[2]), float(block[3])),
                    )
                )

    if not chunks:
        raise PdfValidationError("no extractable text was found")
    return ParsedPaper(
        sha256=sha256,
        title=title,
        chunks=chunks,
        github_urls=sorted(github_urls),
        architecture_pages=sorted(architecture_pages)[:8],
    )


def render_pages(path: Path, pages: list[int], output_dir: Path) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to render PDF pages") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    with fitz.open(path) as document:
        for page_number in pages:
            if page_number < 1 or page_number > document.page_count:
                continue
            output = output_dir / f"page-{page_number:04d}.png"
            document[page_number - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(output)
            rendered.append(output)
    return rendered
