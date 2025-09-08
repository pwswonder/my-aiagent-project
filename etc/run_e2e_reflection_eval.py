# -*- coding: utf-8 -*-
"""
E2E 러너 (안전형)
- 00: 원본 템플릿 저장
- 01a: Jinja 렌더-only
- 01b: 렌더 + SAFE 슬롯 주입
- 두 후보를 analyze_quality(py_src=...)로 평가하여 더 높은 점수를 baseline(01)로 채택
- 03: LangGraph 리플렉션 (없으면 no-op)
- 04: Quality 리플렉션 (없으면 no-op)  ← 시그니처 (template_key, spec, py_src, ...)
- 02_langgraph_eval.json / SUMMARY.json 기록
"""

from __future__ import annotations

# 1) 문제 분석
# - quality_reflection.run_quality_reflection(template_key, spec, py_src, enable=None, max_rounds=1)
# - langgraph_reflection.run_langgraph_reflection(py_src, spec, max_rounds=2)
# - 두 함수 모두 dict로 반환하며 코드 키는 'src' (러너에서 반영 필요)

# 2) 단계별 해결 방안
# - Jinja Undefined + 스펙 기본값 하이드레이션으로 렌더 실패 방지
# - SAFE 슬롯 주입: 의미없는 코드(#TODO/빈/NotImplementedError/pass)면 주입 스킵
# - 리플렉션 래퍼: 위 시그니처에 맞춰 호출, 반환 dict의 'src' 우선 사용
# - 디버그 파일 남김: _render_*_error.txt, _debug_quality_(baseline|final).json

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List


# ---------- 파일 IO 유틸 ----------
def read_text(p: Path) -> str:
    p = Path(p)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def write_text(p: Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def dump_json(p: Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(p: Path) -> Dict[str, Any]:
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 외부 모듈 임포트 ----------
# Jinja2 (렌더-only 후보용)

try:
    from jinja2 import Environment, FileSystemLoader, Undefined
except Exception as e:
    raise RuntimeError("Jinja2가 필요합니다. `pip install jinja2`") from e

# 품질 분석기
try:
    from services.code_quality_analyzer import (
        analyze_quality,
    )  # 반드시 analyze_quality(py_src=..., spec=...) 지원
except Exception as e:
    print(f"[WARN] analyze_quality import failed: {e}")

    def analyze_quality(py_src: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        # 최소 폴백
        return {"score": 10, "issues": [{"code": "NO_ANALYZER"}]}


# 리플렉션 모듈 (없으면 no-op)
try:
    from services.langgraph_reflection import run_langgraph_reflection as _run_langgraph
except Exception:
    _run_langgraph = None

try:
    from services.quality_reflection import run_quality_reflection as _run_quality
except Exception:
    _run_quality = None


def _pylit(value):
    """
    템플릿에서 Python literal이 필요한 곳(예: metrics 리스트)용 필터.
    - dict/list/tuple은 가독성 있는 파이썬 표현으로 변환
    - 나머지는 문자열로 변환
    """
    try:
        from pprint import pformat

        if isinstance(value, (dict, list, tuple, set)):
            return pformat(value, width=88, compact=True)
        return str(value)
    except Exception:
        return str(value)


# ---------- 하이드레이션 ----------
def _hydrate_spec_defaults(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    템플릿에서 참조하는 필수 키가 없을 때 안전 기본값을 채워 렌더 실패를 줄인다.
    """
    s = dict(spec or {})
    dims = dict(s.get("dims") or {})
    training = dict(s.get("training") or {})
    head = dict(s.get("head") or {})
    compile_cfg = dict(s.get("compile") or {})

    # Dims
    dims.setdefault("in_ch", 3)
    dims.setdefault("height", 32)
    dims.setdefault("width", 32)
    dims.setdefault("num_classes", 10)

    # Head
    head.setdefault("units", dims.get("num_classes", 10))
    head.setdefault("activation", "softmax")

    # Compile
    compile_cfg.setdefault("optimizer", "adam")
    compile_cfg.setdefault("loss", "sparse_categorical_crossentropy")
    compile_cfg.setdefault("metrics", ["accuracy"])

    # Training
    training.setdefault("batch_size", 8)
    training.setdefault("epochs", 1)

    s["dims"] = dims
    s["training"] = training
    s["head"] = head
    s["compile"] = compile_cfg
    return s


# ---------- SAFE 슬롯 주입 ----------
_SLOT_LINE = re.compile(
    r"^(?P<indent>\s*)\#\s*"
    r"(?:\{\%\s*raw\s*\%\}\s*)?"
    r"\{\{?\s*CUSTOM_BLOCK:(?P<slot>[^}\s]+)\s*\}\}?"
    r"(?:\s*\{\%\s*endraw\s*\%\})?"
    r"(?:\s*\#.*)?$",
    re.MULTILINE,
)


def _is_meaningful_block(code: str) -> bool:
    """빈/주석/#TODO/NotImplementedError/pass만이면 False."""
    if not code or not isinstance(code, str):
        return False
    lines = [ln for ln in code.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return False
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if first.startswith("# TODO"):
        return False
    body = "\n".join(nonempty)
    if "NotImplementedError" in body:
        return False
    if all(ln.strip() in {"pass", "#", "#.", "#.."} for ln in nonempty):
        return False
    # 의미있는 토큰 휴리스틱
    tokens = [
        "def ",
        "class ",
        "return ",
        "=",
        "(",
        ")",
        "layers.",
        "nn.",
        "tf.",
        "torch.",
        "model.",
        "compile(",
        "fit(",
    ]
    if any(t in body for t in tokens):
        return True
    words = re.findall(r"[A-Za-z_]{4,}", body)
    return len(words) >= 3


def _inject_custom_blocks(rendered_text: str, custom_blocks: Dict[str, str]) -> str:
    """
    SAFE 주입:
    - 의미없는 코드는 주입하지 않고 마커 유지(템플릿 스켈레톤 보호)
    - 들여쓰기 0이면 다음 코드 라인의 들여쓰기 추정
    - 마지막 개행 보장
    """
    if not isinstance(rendered_text, str):
        return rendered_text

    cb = custom_blocks or {}
    snapshot = rendered_text

    def _infer_next_indent(start_idx: int) -> str:
        tail = snapshot[start_idx:]
        for ln in tail.splitlines():
            st = ln.strip()
            if not st or st.startswith("#"):
                continue
            return ln[: len(ln) - len(ln.lstrip())]
        return ""

    def _repl(m: "re.Match[str]") -> str:
        indent = m.group("indent") or ""
        slot = (m.group("slot") or "").strip()
        code = cb.get(slot) or cb.get(slot.lower()) or cb.get(slot.upper()) or ""
        if not _is_meaningful_block(code):
            return m.group(0)
        if indent == "":
            indent = _infer_next_indent(m.end())
        lines = [(indent + ln) if ln.strip() else ln for ln in str(code).splitlines()]
        return "\n".join(lines) + "\n"

    return _slot_normalize_newlines(_SLOT_LINE.sub(_repl, rendered_text))


def _slot_normalize_newlines(text: str) -> str:
    # 슬롯 치환 후 빈 줄이 과하게 붙는 것을 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_custom_blocks(spec: Dict[str, Any]) -> Dict[str, str]:
    """
    spec['custom_blocks'] + spec['blocks'](dict/str) 를 병합.
    동일 키의 대/소문자 변형을 모두 등록하여 케이스 미스 방지.
    """
    out: Dict[str, str] = {}
    cb = (spec or {}).get("custom_blocks", {}) or {}
    if isinstance(cb, dict):
        for k, v in cb.items():
            if not v:
                continue
            out.setdefault(k, v)
            out.setdefault(str(k).upper(), v)
            out.setdefault(str(k).lower(), v)
    blocks = (spec or {}).get("blocks", {}) or {}
    if isinstance(blocks, dict):
        for k, v in blocks.items():
            code = (v.get("code") if isinstance(v, dict) else v) or ""
            if not code:
                continue
            out.setdefault(k, code)
            out.setdefault(str(k).upper(), code)
            out.setdefault(str(k).lower(), code)
    return out


# ---------- 렌더 ----------
# _make_env() 함수 교체 (Undefined 그대로 유지)


def _make_env(template_dir: str) -> Environment:
    loader = FileSystemLoader(template_dir, followlinks=True)
    env = Environment(
        loader=loader,
        undefined=Undefined,  # StrictUndefined 금지
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    # ✅ 커스텀 필터 등록
    env.filters["pylit"] = _pylit
    # 필요 시 여기에 다른 필터도 추가 가능
    return env


# render_only() 교체
def render_only(template_dir: str, tpl_name: str, spec: Dict[str, Any]) -> str:
    env = _make_env(template_dir)
    txt = read_text(Path(template_dir) / tpl_name)
    if re.search(r"{\%\s*for\s+", txt):
        raise ValueError(f"Template {tpl_name} contains a Jinja for-loop (forbidden).")
    tmpl = env.get_template(tpl_name)

    # ✅ 컨텍스트 보강: 최상위 키 보장 + spec 사본 함께 제공
    ctx = _hydrate_spec_defaults(spec)
    ctx.setdefault("spec", ctx)  # 템플릿이 {{ spec.dims... }} 로 접근하는 경우 대비
    ctx.setdefault("dims", ctx.get("dims", {}))
    ctx.setdefault("compile", ctx.get("compile", {}))
    ctx.setdefault("training", ctx.get("training", {}))
    ctx.setdefault("head", ctx.get("head", {}))

    return tmpl.render(**ctx)


def render_and_inject(template_dir: str, tpl_name: str, spec: Dict[str, Any]) -> str:
    code = render_only(template_dir, tpl_name, spec)
    cblocks = build_custom_blocks(spec)
    return _inject_custom_blocks(code, cblocks)


# ---------- 리플렉션 래퍼 ----------
def summarize_eval(result: Dict[str, Any]) -> Tuple[int, int]:
    score = int(result.get("score", 0) or 0) if isinstance(result, dict) else 0
    issues = result.get("issues") if isinstance(result, dict) else None
    ic = (
        len(issues)
        if isinstance(issues, list)
        else int(result.get("issues_count", 0) or result.get("issues_detected", 0) or 0)
    )
    return score, ic


def do_langgraph_reflection(
    py_src: str, spec: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if _run_langgraph is None:
        return py_src, {"note": "langgraph_reflection not available (no-op)"}
    try:
        out = _run_langgraph(py_src=py_src, spec=spec)
    except TypeError:
        out = _run_langgraph(py_src, spec)
    except Exception as e:
        return py_src, {"note": f"langgraph_reflection error: {e}"}
    if isinstance(out, dict):
        code = out.get("src") or out.get("code") or out.get("py_src") or py_src
        return code, out
    return str(out), {}


def do_quality_reflection(
    template_key: str, py_src: str, spec: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if _run_quality is None:
        return py_src, {"note": "quality_reflection not available (no-op)"}
    try:
        # ✅ 올바른 시그니처: (template_key, spec, py_src, enable=None, max_rounds=1)
        out = _run_quality(template_key=template_key, spec=spec, py_src=py_src)
    except TypeError:
        # 구버전 호환: 위치 인자 시도
        try:
            out = _run_quality(template_key, spec, py_src)
        except Exception as e:
            return py_src, {"note": f"quality_reflection call failed: {e}"}
    except Exception as e:
        return py_src, {"note": f"quality_reflection error: {e}"}

    if isinstance(out, dict):
        code = out.get("src") or out.get("code") or out.get("py_src") or py_src
        return code, out
    return str(out), {}


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tpl-root",
        required=True,
        help="templates_scaffolded_slots 또는 templates_scaffolded/services",
    )
    ap.add_argument("--templates", nargs="+", required=True, help="처리할 .j2 목록")
    ap.add_argument("--out-dir", default="./e2e_eval_out")
    ap.add_argument("--gain-threshold", type=int, default=0)
    ap.add_argument("--scenarios-json", required=True)
    args = ap.parse_args()

    tpl_root = Path(args.tpl_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    spec_map = load_json(Path(args.scenarios_json))
    summary_rows: List[Dict[str, Any]] = []

    for tpl_name in args.templates:
        model_key = Path(tpl_name).stem
        raw_spec = spec_map.get(model_key, {}) or {}
        spec = _hydrate_spec_defaults(raw_spec)

        print(f"\n=== [{tpl_name}] processing ===")

        # 출력 경로
        tdir = out_root / model_key
        tdir.mkdir(parents=True, exist_ok=True)
        f00 = tdir / "00_original.j2"
        f01 = tdir / "01_baseline_slot_applied.py"
        f01a = tdir / "01a_render_only.py"
        f01b = tdir / "01b_injected.py"
        f02 = tdir / "02_preseed_for_reflection.py"
        f03 = tdir / "03_after_langgraph.py"
        f04 = tdir / "04_after_quality_reflection.py"
        f02eval = tdir / "02_langgraph_eval.json"

        # 0) 템플릿 저장
        tpl_path = tpl_root / tpl_name
        if not tpl_path.exists():
            warn = f"Template not found: {tpl_path}"
            print("[WARN]", warn)
            dump_json(
                f02eval,
                {
                    "template": tpl_name,
                    "warning": warn,
                    "baseline_score": 0,
                    "final_score": 0,
                    "issues_before": 0,
                    "issues_after": 0,
                    "score_gain": 0,
                    "success": False,
                },
            )
            summary_rows.append(
                {
                    "template": tpl_name,
                    "baseline_score": 0,
                    "final_score": 0,
                    "score_gain": 0,
                    "issues_before": 0,
                    "issues_after": 0,
                    "success": False,
                    "warning": warn,
                }
            )
            continue
        write_text(f00, read_text(tpl_path))

        # 1) 후보 생성
        try:
            cand_a = render_only(str(tpl_root), tpl_name, spec)
        except Exception as e:
            cand_a = ""
            write_text(tdir / "_render_only_error.txt", str(e))
        try:
            cand_b = render_and_inject(str(tpl_root), tpl_name, spec)
        except Exception as e:
            cand_b = ""
            write_text(tdir / "_render_inject_error.txt", str(e))

        write_text(f01a, cand_a)
        write_text(f01b, cand_b)

        # 2) 후보 평가 → 더 높은 점수 채택
        eval_a = (
            analyze_quality(py_src=cand_a, spec=spec)
            if cand_a
            else {"score": 0, "issues": [{"code": "NO_SRC"}]}
        )
        eval_b = (
            analyze_quality(py_src=cand_b, spec=spec)
            if cand_b
            else {"score": 0, "issues": [{"code": "NO_SRC"}]}
        )
        score_a, _ = summarize_eval(eval_a)
        score_b, _ = summarize_eval(eval_b)

        if score_b > score_a:
            baseline_src, base_eval, chosen = cand_b, eval_b, "injected"
        else:
            baseline_src, base_eval, chosen = cand_a, eval_a, "render_only"

        write_text(f01, baseline_src)
        dump_json(
            tdir / "_debug_quality_baseline.json",
            {
                "chosen": chosen,
                "score_a": score_a,
                "score_b": score_b,
                "eval_a": eval_a,
                "eval_b": eval_b,
            },
        )

        # 3) 프리시드
        preseed_src = baseline_src
        write_text(f02, preseed_src)

        # 4) 리플렉션
        code3, lg_report = do_langgraph_reflection(preseed_src, spec)
        write_text(f03, code3)

        code4, q_report = do_quality_reflection(
            model_key, code3, spec
        )  # ✅ template_key 전달
        write_text(f04, code4)

        # 5) 평가
        final_eval = analyze_quality(py_src=code4, spec=spec)
        baseline_score, issues_before = summarize_eval(base_eval)
        final_score, issues_after = summarize_eval(final_eval)
        gain = final_score - baseline_score
        success = gain >= int(args.gain_threshold)

        # 6) 저장
        tpl_report = {
            "template": tpl_name,
            "baseline_score": baseline_score,
            "final_score": final_score,
            "score_gain": gain,
            "issues_before": issues_before,
            "issues_after": issues_after,
            "success": success,
            "chosen": chosen,
            "langgraph_report": lg_report,
            "quality_report": q_report,
        }
        dump_json(f02eval, tpl_report)
        dump_json(tdir / "_debug_quality_final.json", final_eval)

        print(
            f"[RESULT] {tpl_name} | baseline={baseline_score} -> final={final_score} (gain={gain}) | chosen={chosen} | success={success}"
        )

    dump_json(out_root / "SUMMARY.json", summary_rows)
    print("\n=== SUMMARY ===")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
