"""
compare_retrieval_results.py — compare Dense vs RRF retrieval runs

Inputs: two CSVs produced by eval_retrieval(_v2).py (one for 'dense', one for 'rrf').
- The CSV must include columns: qid, bucket, mrr, and a 'hit@k' column (e.g., hit@5).

Outputs:
- comparison_report.md: Overall & per-bucket metrics and significance tests
- (optional) charts: per-bucket bar plots

Usage:
python compare_retrieval_results.py \
  --dense-csv ./eval_out/results_dense_k5.csv \
  --rrf-csv   ./eval_out/results_rrf_k5.csv \
  --outdir    ./eval_out \
  --make-plots
"""

import csv, argparse, os, math, statistics
from collections import defaultdict
import sys


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def detect_cols(rows):
    # find hit@k column name and ndcg@k column
    hit_col = None
    ndcg_col = None
    if not rows:
        return hit_col, ndcg_col
    for c in rows[0].keys():
        if c.lower().startswith("hit@"):
            hit_col = c
        if c.lower().startswith("ndcg@"):
            ndcg_col = c
    # fallback
    if hit_col is None:
        hit_col = "hit@5" if "hit@5" in rows[0] else list(rows[0].keys())[0]
    return hit_col, ndcg_col


def to_float(x):
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def to_int(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0


def pair_by_qid(dense_rows, rrf_rows):
    dmap = {r["qid"]: r for r in dense_rows if "qid" in r}
    rmap = {r["qid"]: r for r in rrf_rows if "qid" in r}
    qids = sorted(set(dmap.keys()) & set(rmap.keys()))
    pairs = [(qid, dmap[qid], rmap[qid]) for qid in qids]
    return pairs


def mcnemar_pvalue(b, c):
    # exact binomial two-sided p-value
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    k = max(b, c)
    # two-sided: sum of tails >= k under Bin(n, 0.5)
    p = 0.0
    for i in range(k, n + 1):
        p += comb(n, i) * (0.5**n)
    p = min(1.0, 2.0 * p)
    return p


def wilcoxon_signed_rank(x, y):
    """
    Returns (W, z, p_approx) using the normal approximation (two-sided).
    """
    # paired diffs
    diffs = [yi - xi for xi, yi in zip(x, y)]

    # remove zeros (no contribution)
    absdiffs, signs = [], []
    for d in diffs:
        if abs(d) > 1e-12:
            absdiffs.append(abs(d))
            signs.append(1 if d > 0 else -1)

    n = len(absdiffs)
    if n == 0:
        return 0.0, 0.0, 1.0

    # rank absolute diffs with average ties
    order = sorted(range(n), key=lambda i: absdiffs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absdiffs[order[j + 1]] == absdiffs[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    Wplus = sum(r for r, s in zip(ranks, signs) if s > 0)
    Wminus = sum(r for r, s in zip(ranks, signs) if s < 0)
    W = min(Wplus, Wminus)

    meanW = n * (n + 1) / 4.0
    varW = n * (n + 1) * (2 * n + 1) / 24.0
    if varW == 0:
        return W, 0.0, 1.0

    z = (W - meanW) / (math.sqrt(varW) + 1e-12)

    # two-sided normal approximation using error function
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return W, z, p


def aggregate_metrics(pairs, hit_col, ndcg_col):
    # pairs: (qid, drow, rrow)
    # collect overall
    hits_d = []
    hits_r = []
    mrr_d = []
    mrr_r = []
    ndcg_d = []
    ndcg_r = []

    buckets = defaultdict(list)

    for qid, d, r in pairs:
        hd = to_int(d.get(hit_col, 0))
        hr = to_int(r.get(hit_col, 0))
        md = to_float(d.get("mrr"))
        mr = to_float(r.get("mrr"))
        nd = to_float(d.get(ndcg_col)) if ndcg_col else None
        nr = to_float(r.get(ndcg_col)) if ndcg_col else None
        hits_d.append(hd)
        hits_r.append(hr)
        mrr_d.append(md)
        mrr_r.append(mr)
        if ndcg_col:
            ndcg_d.append(nd)
            ndcg_r.append(nr)
        b = d.get("bucket") or r.get("bucket") or "general"
        buckets[b].append((hd, hr, md, mr, nd, nr))

    # overall metrics
    N = len(pairs)
    rec_d = sum(hits_d) / N if N else 0.0
    rec_r = sum(hits_r) / N if N else 0.0
    mrr_mean_d = sum(mrr_d) / N if N else 0.0
    mrr_mean_r = sum(mrr_r) / N if N else 0.0
    ndcg_mean_d = (sum(ndcg_d) / N) if (ndcg_col and N) else None
    ndcg_mean_r = (sum(ndcg_r) / N) if (ndcg_col and N) else None

    # McNemar
    b = sum(1 for hd, hr in zip(hits_d, hits_r) if hd == 0 and hr == 1)
    c = sum(1 for hd, hr in zip(hits_d, hits_r) if hd == 1 and hr == 0)
    p_mcn = mcnemar_pvalue(b, c)

    # Wilcoxon on MRR
    W, z, p_wil = wilcoxon_signed_rank(mrr_d, mrr_r)

    # Per-bucket
    bucket_rows = []
    for bucket, items in buckets.items():
        if not items:
            continue
        hd = [it[0] for it in items]
        hr = [it[1] for it in items]
        md = [it[2] for it in items]
        mr = [it[3] for it in items]
        nd = [it[4] for it in items if it[4] is not None]
        nr = [it[5] for it in items if it[5] is not None]
        n = len(items)
        bd = sum(hd) / n
        br = sum(hr) / n
        Wb, zb, pb = wilcoxon_signed_rank(md, mr)
        bb = sum(1 for x, y in zip(hd, hr) if x == 0 and y == 1)
        cc = sum(1 for x, y in zip(hd, hr) if x == 1 and y == 0)
        pmc = mcnemar_pvalue(bb, cc)
        bucket_rows.append(
            {
                "bucket": bucket,
                "n": n,
                "recall_dense": round(bd, 6),
                "recall_rrf": round(br, 6),
                "delta_recall": round(br - bd, 6),
                "mrr_dense": round(sum(md) / n, 6),
                "mrr_rrf": round(sum(mr) / n, 6),
                "delta_mrr": round((sum(mr) - sum(md)) / n, 6),
                "mcnemar_p": round(pmc, 6),
                "wilcoxon_p": round(pb, 6),
            }
        )

    summary = {
        "N": N,
        "hit_col": hit_col,
        "ndcg_col": ndcg_col,
        "overall": {
            "recall_dense": round(rec_d, 6),
            "recall_rrf": round(rec_r, 6),
            "delta_recall": round(rec_r - rec_d, 6),
            "mrr_dense": round(mrr_mean_d, 6),
            "mrr_rrf": round(mrr_mean_r, 6),
            "delta_mrr": round(mrr_mean_r - mrr_mean_d, 6),
            "ndcg_dense": round(ndcg_mean_d, 6) if ndcg_col else None,
            "ndcg_rrf": round(ndcg_mean_r, 6) if ndcg_col else None,
            "delta_ndcg": round((ndcg_mean_r - ndcg_mean_d), 6) if ndcg_col else None,
            "mcnemar_b": b,
            "mcnemar_c": c,
            "mcnemar_p": round(p_mcn, 6),
            "wilcoxon_W": round(W, 6),
            "wilcoxon_z": round(z, 6),
            "wilcoxon_p": round(p_wil, 6),
        },
        "per_bucket": bucket_rows,
    }
    return summary


def save_markdown(summary, outdir):
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "comparison_report.md")
    o = summary["overall"]
    lines = []
    lines.append("# Retrieval Comparison: Dense vs RRF\n")
    lines.append(f"- Samples (N): **{summary['N']}**")
    lines.append(
        f"- Using columns: `{summary['hit_col']}` (hits), `{summary['ndcg_col']}` (nDCG)\n"
    )
    lines.append("## Overall\n")
    lines.append("| Metric | Dense | RRF | Δ | p-value |\n|---|---:|---:|---:|---:|")
    lines.append(
        f"| Recall | {o['recall_dense']:.3f} | {o['recall_rrf']:.3f} | **{o['delta_recall']:.3f}** | McNemar p={o['mcnemar_p']:.3g} |"
    )
    lines.append(
        f"| MRR    | {o['mrr_dense']:.3f} | {o['mrr_rrf']:.3f} | **{o['delta_mrr']:.3f}** | Wilcoxon p={o['wilcoxon_p']:.3g} |"
    )
    if summary["ndcg_col"]:
        lines.append(
            f"| nDCG   | {o['ndcg_dense']:.3f} | {o['ndcg_rrf']:.3f} | **{o['delta_ndcg']:.3f}** | — |"
        )
    lines.append("\n## By Bucket\n")
    lines.append(
        "| bucket | n | Recall(dense) | Recall(rrf) | ΔRecall | MRR(dense) | MRR(rrf) | ΔMRR | McNemar p | Wilcoxon p |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in summary["per_bucket"]:
        lines.append(
            f"| {b['bucket']} | {b['n']} | {b['recall_dense']:.3f} | {b['recall_rrf']:.3f} | {b['delta_recall']:.3f} | "
            f"{b['mrr_dense']:.3f} | {b['mrr_rrf']:.3f} | {b['delta_mrr']:.3f} | {b['mcnemar_p']:.3g} | {b['wilcoxon_p']:.3g} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Wrote markdown: {md_path}")
    return md_path


def make_plots(summary, outdir):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARN] matplotlib not available, skipping plots:", e)
        return []
    os.makedirs(outdir, exist_ok=True)
    paths = []

    # Bar plot per-bucket Recall
    buckets = [b["bucket"] for b in summary["per_bucket"]]
    rd = [b["recall_dense"] for b in summary["per_bucket"]]
    rr = [b["recall_rrf"] for b in summary["per_bucket"]]
    xs = range(len(buckets))

    plt.figure()
    width = 0.35
    plt.bar([x - width / 2 for x in xs], rd, width=width, label="Dense")
    plt.bar([x + width / 2 for x in xs], rr, width=width, label="RRF")
    plt.xticks(list(xs), buckets, rotation=30, ha="right")
    plt.ylabel("Recall")
    plt.title("Recall by Bucket")
    plt.legend()
    p1 = os.path.join(outdir, "recall_by_bucket.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    paths.append(p1)

    # Bar plot per-bucket MRR
    md = [b["mrr_dense"] for b in summary["per_bucket"]]
    mr = [b["mrr_rrf"] for b in summary["per_bucket"]]

    plt.figure()
    plt.bar([x - width / 2 for x in xs], md, width=width, label="Dense")
    plt.bar([x + width / 2 for x in xs], mr, width=width, label="RRF")
    plt.xticks(list(xs), buckets, rotation=30, ha="right")
    plt.ylabel("MRR")
    plt.title("MRR by Bucket")
    plt.legend()
    p2 = os.path.join(outdir, "mrr_by_bucket.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    paths.append(p2)

    print("[OK] Saved plots:", paths)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-csv", required=True)
    ap.add_argument("--rrf-csv", required=True)
    ap.add_argument("--outdir", default="./eval_out")
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    dense_rows = read_csv(args.dense_csv)
    rrf_rows = read_csv(args.rrf_csv)
    if not dense_rows or not rrf_rows:
        print("[FATAL] One of the CSVs is empty or missing rows.")
        sys.exit(1)

    hit_col, ndcg_col = detect_cols(dense_rows)
    pairs = pair_by_qid(dense_rows, rrf_rows)
    if not pairs:
        print("[FATAL] No overlapping qids between the two CSVs.")
        sys.exit(1)

    summary = aggregate_metrics(pairs, hit_col, ndcg_col)
    os.makedirs(args.outdir, exist_ok=True)
    # Save JSON + Markdown
    import json

    json_path = os.path.join(args.outdir, "comparison_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote JSON: {json_path}")
    md_path = save_markdown(summary, args.outdir)
    if args.make_plots:
        make_plots(summary, args.outdir)


if __name__ == "__main__":
    main()
