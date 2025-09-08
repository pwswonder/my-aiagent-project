# eval_retrieval_v2.py
"""
eval_retrieval_v2.py — Retrieval evaluation (PDF/TXT) with robust extraction + diagnostics.

Key additions vs previous:
  - Tries extractors in order: PyPDF → PyMuPDF → pdfminer → (optional) OCR
  - CLI: --min-avg-chars, --force-extractor {pypdf,pymupdf,pdfminer,ocr,auto}, --dpi, --save-stages
  - Prints per-extractor char counts and avg chars/page; saves intermediate text files if requested

Usage:
  python eval_retrieval_v2.py \
    --jsonl gold_attention_is_all_you_need.jsonl \
    --doc /path/to/paper.pdf \
    --source paper.pdf \
    --variant rrf --k 5 --fetchk 40 \
    --ocr --ocr-lang eng+kor --dpi 350 \
    --min-avg-chars 60 \
    --save-stages \
    --save-txt \
    --outdir ./eval_out
"""

import os, sys, json, re, csv, math, random, argparse, statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import project embedder
try:
    from services.embedder import embedder
except Exception as e:
    print("[FATAL] Cannot import services.embedder.embedder:", e)
    sys.exit(1)


def set_env_for_variant(
    variant: str,
    k: int,
    fetchk: int,
    mmr_lambda: float,
    compress: bool,
    weights: str = "0.6,0.4",
):
    os.environ["RAG_BYPASS_CACHE"] = "1"
    os.environ["QA_TOPK"] = str(k)
    os.environ["RAG_FETCHK"] = str(fetchk)
    os.environ["RAG_MMR_LAMBDA"] = str(mmr_lambda)
    os.environ["RAG_DEBUG"] = os.getenv("RAG_DEBUG", "0")
    os.environ["RAG_HYBRID"] = "0" if variant.lower() == "dense" else "1"
    os.environ["RAG_COMPRESS"] = "1" if compress else "0"
    if weights:
        os.environ["RAG_HYBRID_WEIGHTS"] = weights
    
    # eval_retrieval_v2.py, set_env_for_variant() 마지막 줄에 덧붙이기(임시 로그)
    print("[EVAL-ENV]", {k: os.environ.get(k) for k in [
    "RAG_HYBRID","QA_TOPK","RAG_FETCHK","RAG_HYBRID_WEIGHTS",
    "RAG_KEEP_DENSE_M","RAG_COMPRESS" 
    ]})



# ----------- Extractors -----------
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
        import fitz  # PyMuPDF
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
        if "\f" not in txt:
            txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt
    except Exception:
        return ""


def extract_text_from_pdf_ocr(path: str, lang: str = "eng+kor", dpi: int = 300) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception as e:
        print(f"[WARN] OCR requested but pytesseract/pdf2image not available: {e}")
        return ""
    pages_text = []
    try:
        images = convert_from_path(path, dpi=dpi)
    except Exception as e:
        print(
            f"[WARN] pdf2image conversion failed (is poppler installed and in PATH?): {e}"
        )
        return ""
    for img in images:
        try:
            t = pytesseract.image_to_string(img, lang=lang) or ""
        except Exception as e:
            print(f"[WARN] OCR page error: {e}")
            t = ""
        pages_text.append(t.strip())
    return "\f\n".join(pages_text)


def page_stats(text: str) -> Tuple[int, float, int]:
    if not text:
        return 0, 0.0, 0
    pages = text.split("\f") if "\f" in text else [text]
    lens = [len(p.strip()) for p in pages]
    return len(pages), (sum(lens) / max(1, len(lens))), sum(lens)


def choose_text_auto(pypdf_txt: str, pymu_txt: str, miner_txt: str) -> str:
    candidates = [pypdf_txt, pymu_txt, miner_txt]
    return max(candidates, key=lambda t: len(t or "")) or ""


def load_document_text(
    doc_path: str,
    ocr: bool,
    ocr_lang: str,
    dpi: int,
    min_avg_chars: int,
    force: str,
    save_stages: bool,
    outdir: str,
) -> Tuple[str, Dict[str, Any]]:
    ext = os.path.splitext(doc_path)[1].lower()
    diag = {}
    if ext == ".txt":
        with open(doc_path, "r", encoding="utf-8") as f:
            txt = f.read()
        diag["used"] = "txt"
        n, avg, total = page_stats(txt)
        diag["txt"] = {"pages": n, "avg_chars_per_page": avg, "total_chars": total}
        return txt, diag

    if ext != ".pdf":
        raise ValueError(f"Unsupported document type: {ext}")

    def maybe_save(name, content):
        if not save_stages:
            return
        try:
            os.makedirs(outdir, exist_ok=True)
            outp = os.path.join(outdir, f"_stage_{name}.txt")
            with open(outp, "w", encoding="utf-8") as f:
                f.write(content or "")
        except Exception:
            pass

    pypdf_txt = pymu_txt = miner_txt = ocr_txt = ""

    if force in ("auto", "pypdf"):
        pypdf_txt = extract_text_from_pdf_pypdf(doc_path)
        maybe_save("pypdf", pypdf_txt)
        pages, avg, total = page_stats(pypdf_txt)
        diag["pypdf"] = {
            "pages": pages,
            "avg_chars_per_page": avg,
            "total_chars": total,
        }

    if force in ("auto", "pymupdf"):
        pymu_txt = extract_text_from_pdf_pymupdf(doc_path)
        maybe_save("pymupdf", pymu_txt)
        pages, avg, total = page_stats(pymu_txt)
        diag["pymupdf"] = {
            "pages": pages,
            "avg_chars_per_page": avg,
            "total_chars": total,
        }

    if force in ("auto", "pdfminer"):
        miner_txt = extract_text_from_pdf_pdfminer(doc_path)
        maybe_save("pdfminer", miner_txt)
        pages, avg, total = page_stats(miner_txt)
        diag["pdfminer"] = {
            "pages": pages,
            "avg_chars_per_page": avg,
            "total_chars": total,
        }

    best = ""
    if force == "pypdf":
        best = pypdf_txt
    elif force == "pymupdf":
        best = pymu_txt
    elif force == "pdfminer":
        best = miner_txt
    else:
        best = choose_text_auto(pypdf_txt, pymu_txt, miner_txt)

    _, avg_best, total_best = page_stats(best)
    diag["selected_pre_ocr"] = {
        "avg_chars_per_page": avg_best,
        "total_chars": total_best,
    }
    need_ocr = ocr and (avg_best < min_avg_chars)
    diag["need_ocr"] = bool(need_ocr)
    diag["min_avg_chars"] = min_avg_chars

    if need_ocr or force == "ocr":
        ocr_txt = extract_text_from_pdf_ocr(doc_path, lang=ocr_lang, dpi=dpi)
        maybe_save("ocr", ocr_txt)
        pages, avg, total = page_stats(ocr_txt)
        diag["ocr"] = {"pages": pages, "avg_chars_per_page": avg, "total_chars": total}
        if len(ocr_txt) > len(best):
            best = ocr_txt
            diag["used"] = "ocr"

    pages, avg, total = page_stats(best)
    diag["final"] = {"pages": pages, "avg_chars_per_page": avg, "total_chars": total}
    if "used" not in diag:
        used = (
            "pypdf"
            if best == pypdf_txt
            else (
                "pymupdf"
                if best == pymu_txt
                else (
                    "pdfminer"
                    if best == miner_txt
                    else "ocr" if best == ocr_txt else "unknown"
                )
            )
        )
        diag["used"] = used

    return best, diag


# ---------- Eval helpers ----------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def contains_all(text: str, tokens: List[str]) -> bool:
    t = text.lower()
    return all(tok.lower() in t for tok in tokens)


def match_all_regex(text: str, regex_list: List[str]) -> bool:
    if not regex_list:
        return True
    for pat in regex_list:
        try:
            if not re.search(pat, text, re.IGNORECASE | re.DOTALL):
                return False
        except re.error:
            if pat.lower() not in text.lower():
                return False
    return True


def judge_hit(doc_text: str, must: List[str], regexes: List[str]) -> bool:
    if must and not contains_all(doc_text, must):
        return False
    if regexes and not match_all_regex(doc_text, regexes):
        return False
    return True


def dcg_at_k(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(relevances: List[int], k: int) -> float:
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(relevances, k) / denom


def build_retriever(raw_text: str, meta: Dict[str, Any], top_k: int):
    state = {
        "raw_text": raw_text,
        "meta": meta,
        "doc_id": random.randint(100000, 999999),
        "top_k": top_k,
    }
    out = embedder(state)
    if not out or out.get("retriever") is None:
        raise RuntimeError("embedder() returned no retriever.")
    return out["retriever"], out


def evaluate(
    jsonl_path: str,
    doc_path: str,
    source_name: str,
    variant: str,
    k: int,
    fetchk: int,
    mmr_lambda: float,
    compress: bool,
    weights: str,
    outdir: str,
    ocr: bool,
    ocr_lang: str,
    dpi: int,
    save_txt: bool,
    min_avg_chars: int,
    force_extractor: str,
    save_stages: bool,
) -> Dict[str, Any]:

    raw_text, diag = load_document_text(
        doc_path,
        ocr=ocr,
        ocr_lang=ocr_lang,
        dpi=dpi,
        min_avg_chars=min_avg_chars,
        force=force_extractor,
        save_stages=save_stages,
        outdir=outdir,
    )
    print("[DIAG] extraction:", json.dumps(diag, ensure_ascii=False, indent=2))

    if not raw_text or len(raw_text.strip()) < 50:
        raise RuntimeError(
            "Text extraction yielded <50 chars. Check OCR/poppler/tesseract and try --force-extractor ocr --dpi 350."
        )

    meta = {"source": source_name or os.path.basename(doc_path)}
    set_env_for_variant(
        variant,
        k=k,
        fetchk=fetchk,
        mmr_lambda=mmr_lambda,
        compress=compress,
        weights=weights,
    )
    retriever, out_state = build_retriever(raw_text, meta, top_k=k)

    rows = [
        json.loads(l)
        for l in open(jsonl_path, "r", encoding="utf-8").read().splitlines()
        if l.strip()
    ]

    results = []
    hits_binary = []
    mrr_values = []
    ndcg_values = []
    bucket_stats = {}
    for item in rows:
        qid = item["qid"]
        question = item["question"]
        bucket = item.get("bucket", "general")
        gold = item.get("gold", {})
        must = gold.get("must_contain", []) or []
        regexes = gold.get("gold_regex", []) or []
        docs = retriever.get_relevant_documents(question)[:k]

        hit, rank, first_text = 0, None, ""
        relevances = []
        for idx, d in enumerate(docs, start=1):
            txt = normalize_space(getattr(d, "page_content", "") or "")
            ok = judge_hit(txt, must, regexes)
            relevances.append(1 if ok else 0)
            if ok and hit == 0:
                hit, rank, first_text = 1, idx, txt[:240]
        mrr = 1.0 / rank if rank else 0.0
        ndcg = ndcg_at_k(relevances, k)

        results.append(
            {
                "qid": qid,
                "bucket": bucket,
                "variant": variant,
                f"hit@{k}": hit,
                "rank": rank or "",
                "mrr": round(mrr, 6),
                f"ndcg@{k}": round(ndcg, 6),
                "question": question,
                "must": "|".join(must),
                "regex": "|".join(regexes),
                "preview": first_text.replace("\n", " "),
            }
        )
        hits_binary.append(hit)
        mrr_values.append(mrr)
        ndcg_values.append(ndcg)
        bucket_stats.setdefault(
            bucket, {"n": 0, "hits": 0, "mrr_sum": 0.0, "ndcg_sum": 0.0}
        )
        bucket_stats[bucket]["n"] += 1
        bucket_stats[bucket]["hits"] += hit
        bucket_stats[bucket]["mrr_sum"] += mrr
        bucket_stats[bucket]["ndcg_sum"] += ndcg

    N = len(results)
    recall = sum(hits_binary) / N if N else 0.0
    mrr_avg = sum(mrr_values) / N if N else 0.0
    ndcg_avg = sum(ndcg_values) / N if N else 0.0
    bucket_rows = [
        {
            "bucket": b,
            "n": s["n"],
            f"recall@{k}": round(s["hits"] / s["n"] if s["n"] else 0.0, 6),
            "mrr": round(s["mrr_sum"] / s["n"] if s["n"] else 0.0, 6),
            f"ndcg@{k}": round(s["ndcg_sum"] / s["n"] if s["n"] else 0.0, 6),
        }
        for b, s in bucket_stats.items()
    ]

    summary = {
        "variant": variant,
        "k": k,
        "fetchk": fetchk,
        "mmr_lambda": mmr_lambda,
        "compress": compress,
        "weights": weights,
        "N": N,
        f"recall@{k}": round(recall, 6),
        "mrr": round(mrr_avg, 6),
        f"ndcg@{k}": round(ndcg_avg, 6),
        "per_bucket": bucket_rows,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": source_name or os.path.basename(doc_path),
        "jsonl": os.path.abspath(jsonl_path),
        "doc": os.path.abspath(doc_path),
        "extraction_diag": diag,
    }

    os.makedirs(outdir, exist_ok=True)
    tag = f"{variant}_k{k}"
    csv_path = os.path.join(outdir, f"results_{tag}.csv")
    sum_path = os.path.join(outdir, f"summary_{tag}.json")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as _csv

            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_txt:
        txt_out = os.path.join(
            outdir,
            os.path.splitext(os.path.basename(doc_path))[0] + ".extracted.final.txt",
        )
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[OK] Saved extracted text: {txt_out}")

    print(f"[OK] CSV: {csv_path}")
    print(f"[OK] Summary: {sum_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, csv_path, sum_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to golden set JSONL")
    ap.add_argument("--doc", required=True, help="Path to *PDF or TXT* document")
    ap.add_argument("--source", default="", help="Source name (e.g., filename.pdf)")
    ap.add_argument("--variant", choices=["dense", "rrf"], default="rrf")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fetchk", type=int, default=40)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--weights", default="0.6,0.4")
    ap.add_argument("--outdir", default="./eval_out")
    ap.add_argument(
        "--ocr", action="store_true", help="Enable OCR fallback for scanned PDFs"
    )
    ap.add_argument("--ocr-lang", default="eng+kor")
    ap.add_argument("--dpi", type=int, default=300, help="OCR DPI (default 300)")
    ap.add_argument("--save-txt", action="store_true", help="Save final extracted text")
    ap.add_argument(
        "--save-stages",
        action="store_true",
        help="Save intermediate texts from each extractor",
    )
    ap.add_argument(
        "--min-avg-chars",
        type=int,
        default=100,
        help="Min avg chars/page to trigger OCR",
    )
    ap.add_argument(
        "--force-extractor",
        choices=["auto", "pypdf", "pymupdf", "pdfminer", "ocr"],
        default="auto",
        help="Force a specific extractor (or auto)",
    )
    args = ap.parse_args()

    evaluate(
        jsonl_path=args.jsonl,
        doc_path=args.doc,
        source_name=args.source,
        variant=args.variant,
        k=args.k,
        fetchk=args.fetchk,
        mmr_lambda=args.mmr_lambda,
        compress=args.compress,
        weights=args.weights,
        outdir=args.outdir,
        ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        dpi=args.dpi,
        save_txt=args.save_txt,
        min_avg_chars=args.min_avg_chars,
        force_extractor=args.force_extractor,
        save_stages=args.save_stages,
    )


if __name__ == "__main__":
    main()
