# extractor_diagnose.py
"""
extractor_diagnose.py — quick check of PDF text extraction stack.

It tries PyPDF, PyMuPDF, pdfminer, and (optionally) OCR; prints page counts and char stats.
"""

import os, json, argparse, re


def page_stats(text: str):
    if not text:
        return 0, 0.0, 0
    pages = text.split("\f") if "\f" in text else [text]
    lens = [len(p.strip()) for p in pages]
    return len(pages), sum(lens) / max(1, len(lens)), sum(lens)


def extract_text_from_pdf_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        r = PdfReader(path)
        pages = []
        for p in r.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t.strip())
        return "\f\n".join(pages)
    except Exception:
        return ""


def extract_text_from_pdf_pymupdf(path: str) -> str:
    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(path)
        pages = []
        for p in doc:
            t = p.get_text("text") or ""
            pages.append(t.strip())
        return "\f\n".join(pages)
    except Exception:
        return ""


def extract_text_from_pdf_pdfminer(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""
    try:
        txt = extract_text(path) or ""
        return txt
    except Exception:
        return ""


def extract_text_from_pdf_ocr(path: str, lang="eng+kor", dpi=300) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return ""
    try:
        images = convert_from_path(path, dpi=dpi)
    except Exception:
        return ""
    pages = []
    for img in images:
        try:
            t = pytesseract.image_to_string(img, lang=lang) or ""
        except Exception:
            t = ""
        pages.append(t.strip())
    return "\f\n".join(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr-lang", default="eng+kor")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--outdir", default="./diag_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    stats = {}
    for name, fn in [
        ("pypdf", extract_text_from_pdf_pypdf),
        ("pymupdf", extract_text_from_pdf_pymupdf),
        ("pdfminer", extract_text_from_pdf_pdfminer),
    ]:
        txt = fn(args.pdf)
        with open(os.path.join(args.outdir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(txt or "")
        n, avg, total = page_stats(txt)
        stats[name] = {"pages": n, "avg_chars_per_page": avg, "total_chars": total}

    if args.ocr:
        ocr_txt = extract_text_from_pdf_ocr(args.pdf, lang=args.ocr_lang, dpi=args.dpi)
        with open(os.path.join(args.outdir, f"ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_txt or "")
        n, avg, total = page_stats(ocr_txt)
        stats["ocr"] = {"pages": n, "avg_chars_per_page": avg, "total_chars": total}

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
