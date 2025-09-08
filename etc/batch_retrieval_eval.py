#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_retrieval_eval.py
- 여러 RRF/Dense 시나리오를 한 번에 실행하고, 요약 표를 합쳐서 저장합니다.
- eval_retrieval_v2.evaluate()를 직접 호출하므로 zsh 인자 처리 이슈가 없습니다.
"""
import os, json, argparse, datetime, pathlib
from typing import List, Dict, Any, Tuple

# 평가 함수 import (당신의 eval_retrieval_v2.py 기준)
from eval_retrieval_v2 import evaluate

# ====== 시나리오 정의 ======
# name: 결과 폴더명/표시명
# variant: "dense" or "rrf"
# fetchk: 후보군 크기
# weights: RRF 가중치 "dense,bm25" (dense일 때는 무시)
# keep_m: Dense 상위 m개 강제 보존 (embedder.py의 safe-keep 패치 사용 시)
# compress: True면 EmbeddingsFilter 압축 on (평가에선 보통 False 권장)
# force: (선택) embedder의 RAG_FORCE 스위치 사용 시 "dense","bm25","rrf" 중 선택, 아니면 "".
SCENARIOS = [
    {
        "name": "dense_f60",
        "variant": "dense",
        "fetchk": 60,
        "weights": "",
        "keep_m": 0,
        "compress": False,
        "force": "",
    },
    {
        "name": "rrf_w80_f60_km2",
        "variant": "rrf",
        "fetchk": 60,
        "weights": "0.8,0.2",
        "keep_m": 2,
        "compress": False,
        "force": "",
    },
    {
        "name": "rrf_w70_f60_km2",
        "variant": "rrf",
        "fetchk": 60,
        "weights": "0.7,0.3",
        "keep_m": 2,
        "compress": False,
        "force": "",
    },
    {
        "name": "rrf_w80_f60_km1",
        "variant": "rrf",
        "fetchk": 60,
        "weights": "0.8,0.2",
        "keep_m": 1,
        "compress": False,
        "force": "",
    },
    {
        "name": "rrf_w80_f60_km3",
        "variant": "rrf",
        "fetchk": 60,
        "weights": "0.8,0.2",
        "keep_m": 3,
        "compress": False,
        "force": "",
    },
    {
        "name": "rrf_w80_f80_km2",
        "variant": "rrf",
        "fetchk": 80,
        "weights": "0.8,0.2",
        "keep_m": 2,
        "compress": False,
        "force": "",
    },
    # 필요하면 다음도 추가:
    # {"name": "force_dense_only", "variant": "rrf", "fetchk": 60, "weights": "0.8,0.2","keep_m": 0, "compress": False, "force": "dense"},
    # {"name": "force_bm25_only", "variant": "rrf", "fetchk": 60, "weights": "0.8,0.2","keep_m": 0, "compress": False, "force": "bm25"},
]


def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def run_one(
    args, scen: Dict[str, Any], batch_root: str
) -> Tuple[Dict[str, Any], str, str]:
    """
    단일 시나리오 실행 → (summary_dict, csv_path, summary_path)
    """
    # 캐시 우회 + 디버그
    os.environ["RAG_BYPASS_CACHE"] = "1"
    os.environ["RAG_DEBUG"] = os.getenv(
        "RAG_DEBUG", "1"
    )  # 기본 1로 켭니다(로그 확인용)

    # safe-keep / force 스위치
    os.environ["RAG_KEEP_DENSE_M"] = str(scen.get("keep_m", 0))
    if scen.get("force"):
        os.environ["RAG_FORCE"] = scen["force"]
    else:
        os.environ.pop("RAG_FORCE", None)  # 이전 값 제거

    # 개별 outdir
    outdir = os.path.join(batch_root, scen["name"])
    ensure_dir(outdir)

    # 공통 인자
    common_kwargs = dict(
        jsonl_path=args.jsonl,
        doc_path=args.doc,
        source_name=args.source or os.path.basename(args.doc),
        variant=scen["variant"],
        k=args.k,
        fetchk=scen["fetchk"],
        mmr_lambda=args.mmr_lambda,
        compress=bool(scen["compress"]),
        weights=scen["weights"] or "0.6,0.4",
        outdir=outdir,
        ocr=bool(args.ocr),
        ocr_lang=args.ocr_lang,
        dpi=args.dpi,
        save_txt=bool(args.save_txt),
        min_avg_chars=args.min_avg_chars,
        force_extractor=args.force_extractor,
        save_stages=bool(args.save_stages),
    )

    # 콘솔 표시
    print(f"\n=== [RUN] {scen['name']} ===")
    print("[ENV] RAG_KEEP_DENSE_M=", os.environ.get("RAG_KEEP_DENSE_M"))
    if "RAG_FORCE" in os.environ:
        print("[ENV] RAG_FORCE=", os.environ["RAG_FORCE"])
    print(
        "[PARAMS]",
        {
            k: common_kwargs[k]
            for k in ["variant", "fetchk", "weights", "compress", "outdir"]
        },
    )

    summary, csv_path, sum_path = evaluate(**common_kwargs)
    return summary, csv_path, sum_path


def make_combined_reports(
    records: List[Tuple[str, Dict[str, Any], str, str]], batch_root: str, k: int
) -> None:
    """
    records: [(scenario_name, summary_dict, csv_path, summary_path), ...]
    """
    # baseline(첫 dense) 찾기
    baseline = None
    for name, summ, _, _ in records:
        if summ.get("variant") == "dense" and baseline is None:
            baseline = (name, summ)
    # 표용 데이터 만들기
    rows = []
    for name, summ, csvp, sump in records:
        row = {
            "scenario": name,
            "variant": summ.get("variant"),
            "N": summ.get("N"),
            f"recall@{k}": summ.get(f"recall@{k}", 0.0),
            "mrr": summ.get("mrr", 0.0),
            f"ndcg@{k}": summ.get(f"ndcg@{k}", 0.0),
            "fetchk": summ.get("fetchk"),
            "weights": summ.get("weights"),
            "compress": summ.get("compress"),
            "csv": csvp,
            "summary": sump,
        }
        rows.append(row)

    # 델타 계산
    if baseline:
        bname, bsum = baseline
        bref = {
            "recall": bsum.get(f"recall@{k}", 0.0),
            "mrr": bsum.get("mrr", 0.0),
            "ndcg": bsum.get(f"ndcg@{k}", 0.0),
        }
        for r in rows:
            r["Δ_recall"] = round(r[f"recall@{k}"] - bref["recall"], 6)
            r["Δ_mrr"] = round(r["mrr"] - bref["mrr"], 6)
            r["Δ_ndcg"] = round(r[f"ndcg@{k}"] - bref["ndcg"], 6)

    # 저장
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    combined_csv = os.path.join(batch_root, f"_combined_{stamp}.csv")
    combined_md = os.path.join(batch_root, f"_combined_{stamp}.md")
    # CSV
    import csv

    if rows:
        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Markdown
    def md_table(rows: List[Dict[str, Any]]) -> str:
        cols = [
            "scenario",
            "variant",
            "N",
            f"recall@{k}",
            "mrr",
            f"ndcg@{k}",
            "Δ_recall",
            "Δ_mrr",
            "Δ_ndcg",
            "fetchk",
            "weights",
            "compress",
        ]
        out = []
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(out)

    best_recall = max(rows, key=lambda r: r.get(f"recall@{k}", 0.0))
    best_mrr = max(rows, key=lambda r: r.get("mrr", 0.0))
    best_ndcg = max(rows, key=lambda r: r.get(f"ndcg@{k}", 0.0))
    with open(combined_md, "w", encoding="utf-8") as f:
        f.write(f"# Batch Retrieval Evaluation (k={k})\n\n")
        f.write(f"- Total Scenarios: **{len(rows)}**\n")
        if baseline:
            f.write(
                f"- Baseline: **{baseline[0]}** (variant={baseline[1].get('variant')})\n"
            )
        f.write(
            f"- Best Recall: **{best_recall['scenario']}** ({best_recall[f'recall@{k}']})\n"
        )
        f.write(f"- Best MRR: **{best_mrr['scenario']}** ({best_mrr['mrr']})\n")
        f.write(
            f"- Best nDCG: **{best_ndcg['scenario']}** ({best_ndcg[f'ndcg@{k}']})\n\n"
        )
        f.write(md_table(rows))
        f.write("\n")

    print("\n[OK] Combined CSV:", combined_csv)
    print("[OK] Combined Markdown:", combined_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--outdir", default="./eval_out_batch")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr-lang", default="eng+kor")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--save-txt", action="store_true")
    ap.add_argument("--save-stages", action="store_true")
    ap.add_argument("--min-avg-chars", type=int, default=100)
    ap.add_argument(
        "--force-extractor",
        choices=["auto", "pypdf", "pymupdf", "pdfminer", "ocr"],
        default="pymupdf",
    )
    args = ap.parse_args()

    # 배치 루트 폴더
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    batch_root = os.path.join(args.outdir, f"batch_{stamp}")
    ensure_dir(batch_root)

    records = []
    for scen in SCENARIOS:
        summ, csvp, sump = run_one(args, scen, batch_root)
        records.append((scen["name"], summ, csvp, sump))

    make_combined_reports(records, batch_root, args.k)


if __name__ == "__main__":
    main()
