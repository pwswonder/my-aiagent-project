# -*- coding: utf-8 -*-
"""
실험 재현 스크립트: 템플릿 스모크(PASS)와 코드 정적 점검 통과율
- 프로젝트 루트에서 실행을 권장합니다(services/* import 가능해야 함).
- Python 3.10+, Jinja2 필요 (pip install jinja2). TensorFlow 설치는 선택(compile 실행 X, ast/compile만 사용).
- 절대 임의 수치를 생성하지 않습니다. 각 템플릿을 실제 렌더링한 결과에 대해만 판단합니다.

PASS 기준(우리 실험 정의)
1) 템플릿 스모크: (a) Jinja 렌더 성공 AND (b) 문법 프리플라이트(compile) OK
2) 코드 정적 점검 통과: code_quality_analyzer.analyze_quality() 결과 score >= 90

산출물
- ./results/smoke_static_results.csv
- ./results/summaries/{template}.txt (요약/이슈)
"""

import os, json, csv, traceback
from typing import Dict, Any, List, Tuple

# 프로젝트 모듈
from services.codegen import render_model_source
from services.codegen_autoblocks import autofill_custom_blocks
from services.quality_reflection import syntax_preflight
from services.code_quality_analyzer import analyze_quality

# 템플릿 탐색: manifest 우선, 없으면 services/templates/*.j2
MANIFEST_PATH = os.path.join("services", "templates_manifest.json")
TEMPLATE_DIR = os.path.join("services", "templates")


def load_templates() -> List[str]:
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = []
        for it in data:
            fname = os.path.splitext(os.path.basename(it.get("file") or ""))[0]
            if fname:
                names.append(fname)
        return sorted(set(names))
    else:
        names = []
        for fn in os.listdir(TEMPLATE_DIR):
            if fn.endswith(".j2"):
                names.append(os.path.splitext(fn)[0])
        return sorted(set(names))


# family별 최소 스펙 시드
def seed_spec_for(template_key: str) -> Dict[str, Any]:
    key = template_key.lower()
    spec: Dict[str, Any] = {
        "proposed_model_family": key,  # family 신호
        "task_type": "",  # 필요 시 설정
        "subtype": "",
        "modality": "",
        "input_shape": [32, 32, 3],
        "num_classes": 10,
        "optimizer_name": "adam",
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "custom_blocks": {},  # 슬롯 초기값
        "evidence": [],
        "title": "",
        "notes": "",
        "baselines": [],
    }
    # 태스크/모달리티 합리적 기본값(템플릿별)
    if "transformer_mt" in key:
        spec.update(
            {
                "task_type": "machine_translation",
                "modality": "text",
                "subtype": "encoderdecoder",
            }
        )
        spec["input_shape"] = [128]  # 토큰 길이
        spec["num_classes"] = 32000  # vocab size
    elif key in ("transformer", "swin", "rnn_seq"):
        spec.update({"task_type": "classification", "modality": "text"})
        spec["input_shape"] = [128]
        spec["num_classes"] = 8
    elif key in ("resnet", "cnn_family", "unet", "gan", "vae", "autoencoder"):
        spec.update({"task_type": "classification", "modality": "image"})
        spec["input_shape"] = [64, 64, 3]
        spec["num_classes"] = 10
        if key == "unet":
            spec.update({"task_type": "segmentation"})
            spec["num_classes"] = 3
        if key == "gan":
            spec["loss"] = "binary_crossentropy"
            spec["metrics"] = []
        if key == "autoencoder":
            spec["loss"] = "mse"
            spec["metrics"] = []
    return spec


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    out_dir = os.path.join(".", "results")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "summaries"))

    templates = load_templates()
    rows = []
    for tkey in templates:
        rec = {
            "template": tkey,
            "render_ok": False,
            "syntax_ok": False,
            "score": None,
            "pass_smoke": False,
            "pass_static": False,
            "issues_high": 0,
            "issues_med": 0,
            "issues_low": 0,
            "error": "",
        }
        try:
            spec = seed_spec_for(tkey)
            # AutoBlocks로 필수 슬롯을 채워 spec.custom_blocks 생성
            spec = autofill_custom_blocks(spec, tkey)

            # 1) 렌더링
            py_src = render_model_source(tkey, spec)
            rec["render_ok"] = isinstance(py_src, str) and len(py_src) > 0

            # 2) 문법 프리플라이트
            ok, log = syntax_preflight(py_src)
            rec["syntax_ok"] = ok

            # 3) 정적 점검(스코어/이슈)
            report = analyze_quality(py_src, spec)
            rec["score"] = report.get("score")
            # 이슈 집계
            hi = sum(1 for it in report["issues"] if it.get("severity") == "high")
            md = sum(1 for it in report["issues"] if it.get("severity") == "medium")
            lo = sum(1 for it in report["issues"] if it.get("severity") == "low")
            rec.update({"issues_high": hi, "issues_med": md, "issues_low": lo})

            # PASS 판정
            rec["pass_smoke"] = bool(rec["render_ok"] and rec["syntax_ok"])
            rec["pass_static"] = bool(
                rec["syntax_ok"]
                and isinstance(rec["score"], int)
                and rec["score"] >= 90
            )

            # 요약 파일
            with open(
                os.path.join(out_dir, "summaries", f"{tkey}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(
                    f"[{tkey}] render_ok={rec['render_ok']} syntax_ok={rec['syntax_ok']} score={rec['score']}\n"
                )
                f.write("\n== issues ==\n")
                for it in report["issues"]:
                    f.write(
                        f"- ({it.get('severity')}) {it.get('code')}: {it.get('msg')}\n  hint: {it.get('hint')}\n"
                    )

        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            with open(
                os.path.join(out_dir, "summaries", f"{tkey}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(f"[{tkey}] ERROR: {rec['error']}\n")
                f.write(traceback.format_exc())

        rows.append(rec)

    # --- 결과 저장 (안전 버전) ---

    # 0) 결과 디렉터리와 CSV 경로를 루프보다 "먼저" 고정 선언
    out_dir = os.path.join(".", "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "smoke_static_results.csv")

    # 1) rows가 비어 있어도 동작하도록 컬럼을 고정
    fieldnames = [
        "template",
        "render_ok",
        "syntax_ok",
        "score",
        "pass_smoke",
        "pass_static",
        "issues_high",
        "issues_med",
        "issues_low",
        "error",  # 에러는 항상 컬럼 포함
    ]

    # 2) DictWriter: extrasaction='ignore' 로 여분 키 무시
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r.setdefault("error", "")  # error 키가 없으면 빈칸으로
            w.writerow(r)

    # 3) 요약 출력(빈 rows도 안전)
    total = len(rows)
    smoke_pass = sum(1 for r in rows if r.get("pass_smoke"))
    static_pass = sum(1 for r in rows if r.get("pass_static"))
    print(
        f"[SUMMARY] templates={total}, smoke_pass={smoke_pass}, static_pass={static_pass}"
    )
    print(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
