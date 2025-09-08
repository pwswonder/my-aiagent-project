import streamlit as st
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import os

# [ADDED] Base code 아티팩트 섹션 표시용 import (표준 라이브러리만)
from pathlib import Path
import json
import contextlib


# === Progress UI & Background Helpers (NO SPINNER) ===
# 이 블록은 app.py 최상단, 모든 사용처보다 '먼저' 정의되어 있어야 합니다.
import threading
import queue
import time

# === 진행 단계 엔진 (Phase -> PHASES -> helpers) ===============================


@dataclass
class Phase:
    name: str
    start_pct: float  # inclusive
    end_pct: float  # exclusive


# 단일 소스: 진행 단계 정의(절대 퍼센트)
PHASES: list[Phase] = [
    Phase("📄 PDF 파싱", 0.0, 30.0),
    Phase("🧪 스펙 생성", 30.0, 55.0),
    Phase("🧭 라우팅", 55.0, 70.0),
    Phase("🤖 LLM 코드 생성", 70.0, 92.0),
    Phase("📝 파일 쓰기", 92.0, 98.0),
    Phase("🔁 결과 반영", 98.0, 99.5),
    Phase("🖥️ 화면 갱신", 99.5, 100.01),
]


def _render_phase_timeline(current_pct: float, phases: list[Phase]) -> str:
    """
    현재 진행률에 따라 타임라인을 Markdown으로 렌더링.
    완료:[x], 진행중:[>], 대기:[ ]
    """
    # 현재 단계 인덱스 계산 (실수 퍼센트 기준)
    now_idx = max(
        0,
        min(
            len(phases) - 1,
            next(
                (i for i, p in enumerate(phases) if current_pct < p.end_pct),
                len(phases) - 1,
            ),
        ),
    )
    lines = ["### 🔄 진행 단계"]
    for i, ph in enumerate(phases):
        if current_pct >= ph.end_pct:
            prefix = "[x]"
        elif i == now_idx:
            prefix = "[>]"
        else:
            prefix = "[ ]"
        lines.append(f"{prefix} {ph.name}")
    lines.append(f"\n**현재 단계:** {phases[now_idx].name}")
    return "\n".join(lines)


# def _smooth_wait_with_substeps(
#     done_evt: "threading.Event",
#     steps_box,  # st.empty()
#     progress,  # st.progress(...)
#     min_pct: int = 30,
#     max_pct: int = 99,
#     max_secs: int = 150,
# ) -> int:
#     """
#     - 진행률(정수 퍼센트)은 progress 바에만 사용
#     - 단계 판단은 '실수 퍼센트'로 하고, 전역 PHASES를 단일 소스로 사용
#     - 완료 시 98%/99.5%/100%를 강제로 밟으며 라벨을 반드시 보여줌
#     """
#     t0 = time.time()
#     pct = float(min_pct)

#     def _tick(cur_pct: float, text: str):
#         progress.progress(int(min(100, cur_pct)), text=text)
#         steps_box.markdown(_render_phase_timeline(cur_pct, PHASES))

#     # 스무딩 루프
#     while not done_evt.is_set():
#         elapsed = time.time() - t0
#         frac = min(1.0, elapsed / max_secs)
#         eased = 1 - (1 - frac) * (1 - frac)  # quadratic ease-out
#         pct = float(min_pct) + eased * float(max_pct - min_pct)
#         pct = max(float(min_pct), min(float(max_pct), pct))
#         _tick(pct, text=f"서버 분석 중... (경과 {int(elapsed)}s)")
#         time.sleep(0.2)

#     # 완료 시 최종 단계들을 '가시적으로' 통과
#     for hard, msg in [
#         (98.0, "🔁 결과 반영 중..."),
#         (99.6, "🖥️ 화면 갱신 중..."),
#         (100.0, "완료"),
#     ]:
#         pct = max(pct, hard)
#         _tick(pct, text=msg)
#         time.sleep(0.12)

#     return int(min(100, pct))

# ===============================================================================


# === Unicode Spinner (크기/속도/스타일 조절 가능) ==============================
# === Unicode Spinner (크기/속도/스타일 조절 + 별칭/우선순위 지원) =============
def _spinner_frame(
    fps: int | None = None,
    style: str | None = None,
    size_px: int | None = None,
) -> str:
    """
    반환: HTML <span> (unsafe_allow_html=True로 렌더)
    우선순위: 함수 인자 > st.session_state > 환경변수 > 기본값
    - fps: 초당 프레임 수 (기본 12)   → 커질수록 빨라짐
    - style: 'braille' | 'quarters' | 'bars' | 'dots' + 별칭('bar','block','quarter')
    - size_px: 글리프 폰트 크기(px), 기본 22
    """
    import os, time
    import streamlit as st

    # ---- 구성값 해석 (우선순위 적용) ----
    ui_style = (st.session_state.get("SPINNER_STYLE_UI") or "").strip().lower()
    env_style = (os.getenv("SPINNER_STYLE", "")).strip().lower()
    style = (style or ui_style or env_style or "braille").strip().lower()

    # 별칭 매핑(오타/단수 처리)
    alias = {
        "bar": "bars",
        "block": "bars",
        "quarter": "quarters",
        "quarterly": "quarters",
    }
    style = alias.get(style, style)

    # 속도/크기도 동일한 우선순위
    ui_fps = st.session_state.get("SPINNER_FPS_UI")
    ui_size = st.session_state.get("SPINNER_SIZE_UI")
    try:
        fps = int(
            (
                fps
                if fps is not None
                else (ui_fps if ui_fps is not None else os.getenv("SPINNER_FPS", 12))
            )
        )
    except Exception:
        fps = 12
    try:
        size_px = int(
            (
                size_px
                if size_px is not None
                else (
                    ui_size if ui_size is not None else os.getenv("SPINNER_SIZE_PX", 22)
                )
            )
        )
    except Exception:
        size_px = 22

    # ---- 프레임 세트 ----
    frames_map: dict[str, list[str]] = {
        "braille": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "quarters": ["◴", "◷", "◶", "◵"],
        "bars": ["▁", "▃", "▄", "▅", "▆", "▇", "▆", "▅", "▄", "▃"],
        "dots": ["∙", "•", "●", "•"],
    }
    frames = frames_map.get(style, frames_map["braille"])

    # ---- 현재 프레임 계산 (fps 기반) ----
    idx = int(time.time() * max(1, fps)) % len(frames)
    glyph = frames[idx]

    # ---- 크게 보이도록 폰트 크기 지정 ----
    return f'<span style="font-size:{int(size_px)}px; display:inline-block; line-height:1">{glyph}</span>'


# === CSS 1회 주입 (스피너 포함) ===============================================
def _inject_phase_css_once():
    """
    단계 리스트/카드용 CSS를 1회만 주입 (스피너 포함).
    """
    key = "_phase_css_injected"
    if st.session_state.get(key):
        return
    st.session_state[key] = True

    st.markdown(
        """
        <style>
        .phase-list { list-style: none; padding-left: 0; margin: 6px 0 0 0; }
        .phase-li { display: flex; align-items: center; gap: 8px; padding: 6px 4px; }
        .phase-name { font-weight: 600; color: #e5e7eb; }   /* 다크테마 대비 */
        .phase-sub  { font-size: 12px; color: #9ca3af; margin-left: 4px; }

        .phase-ico { min-width: 32px; text-align: center; display: inline-block; }

        /* 현재 단계 스피너 */
        .spin-dot {
            display: inline-block;
            width: 14px; height: 14px;
            border: 2px solid #475569;      /* slate-600 */
            border-top-color: #60a5fa;      /* blue-400 */
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            vertical-align: -2px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


# === 체크리스트 HTML 렌더러 (진행중: 스피너) ================================
def _render_phase_timeline_html(
    current_pct: float,
    phases: list[Phase],
    mode: str = "list",
    spinner_glyph: str | None = None,  # ★ 추가: 현재 단계에 표시할 스피너 문자
) -> str:
    """
    단계 렌더링(HTML). mode='list': 체크리스트형(완료=✅, 진행중=스피너, 대기=☐)
    """
    now_idx = max(
        0,
        min(
            len(phases) - 1,
            next(
                (i for i, p in enumerate(phases) if current_pct < p.end_pct),
                len(phases) - 1,
            ),
        ),
    )

    if mode == "list":
        items = []
        for i, ph in enumerate(phases):
            if current_pct >= ph.end_pct:
                ico_html = '<span class="phase-ico">✅</span>'
                sub = '<span class="phase-sub">완료</span>'
            elif i == now_idx:
                # ★ CSS 없이도 도는 스피너(유니코드 프레임)를 사용
                ico = spinner_glyph or "⟳"
                ico_html = f'<span class="phase-ico">{ico}</span>'
                sub = '<span class="phase-sub">진행중</span>'
            else:
                ico_html = '<span class="phase-ico">☐</span>'
                sub = '<span class="phase-sub">대기</span>'
            items.append(
                f'<li class="phase-li">{ico_html}<span class="phase-name">{ph.name}</span>{sub}</li>'
            )
        return '<ul class="phase-list">' + "".join(items) + "</ul>"

    # (cards 모드 필요시 기존 구현 유지)
    return (
        '<ul class="phase-list">'
        + "".join(
            f'<li class="phase-li"><span class="phase-ico">☐</span><span class="phase-name">{ph.name}</span></li>'
            for ph in phases
        )
        + "</ul>"
    )


# === 스무딩 함수(HTML로 렌더) ===============================================
def _smooth_wait_with_substeps(
    done_evt: "threading.Event",
    steps_box,
    progress,
    min_pct: int = 30,
    max_pct: int = 99,
    max_secs: int = 150,
) -> int:
    t0 = time.time()
    pct = float(min_pct)

    def _tick(cur_pct: float, text: str, spinning: bool = True):
        # 진행률 업데이트(정수 퍼센트)
        progress.progress(int(min(100, cur_pct)), text=text)
        # ★ 현재 단계 스피너 프레임 전달
        html = _render_phase_timeline_html(
            cur_pct,
            PHASES,
            mode="list",
            spinner_glyph=_spinner_frame() if spinning else None,
        )
        steps_box.markdown(html, unsafe_allow_html=True)

    # 스무딩 루프(진행중엔 계속 스피너 회전)
    while not done_evt.is_set():
        elapsed = time.time() - t0
        frac = min(1.0, elapsed / max_secs)
        eased = 1 - (1 - frac) * (1 - frac)
        pct = float(min_pct) + eased * float(max_pct - min_pct)
        pct = max(float(min_pct), min(float(max_pct), pct))
        _tick(pct, text=f"서버 분석 중... (경과 {int(elapsed)}s)", spinning=True)
        time.sleep(0.2)

    # 완료 구간(스피너 정지 → 체크 표시가 보이도록 spinning=False)
    for hard, msg in [
        (98.0, "🔁 결과 반영 중..."),
        (99.6, "🖥️ 화면 갱신 중..."),
        (100.0, "완료"),
    ]:
        pct = max(pct, hard)
        _tick(pct, text=msg, spinning=False)
        time.sleep(0.12)

    return int(min(100, pct))


# ===============================================================================


def _render_steps(current_idx: int, labels: list[str]) -> str:
    """
    진행 단계 목록을 아이콘과 함께 마크다운으로 렌더링합니다.
    - 완료: ✅, 진행중: 🔄, 대기: ⏳
    """
    lines: list[str] = []
    for i, label in enumerate(labels):
        if i < current_idx:
            prefix = "✅"
        elif i == current_idx:
            prefix = "🔄"
        else:
            prefix = "⏳"
        lines.append(f"- {prefix} **{label}**")
    return "\n".join(lines)


def _start_analysis_job_bytes(
    file_bytes: bytes,
    filename: str,
    fastapi_url: str,
    done_evt: "threading.Event",
    out_q: "queue.Queue",
):
    """
    /documents/analyze_only 를 '백그라운드 쓰레드'에서 호출합니다.
    - 성공 시: out_q.put(result: dict)
    - 실패 시: out_q.put(("__error__", str(e)))
    - 언제든 종료 시: done_evt.set()
    주의: 쓰레드 안에서는 st.session_state 를 절대 건드리지 않습니다.
    """
    import requests

    def _worker():
        try:
            files = {"file": (filename, file_bytes, "application/pdf")}
            resp = requests.post(
                f"{fastapi_url}/documents/analyze_only", files=files, timeout=600
            )
            resp.raise_for_status()
            out_q.put(resp.json())
        except Exception as e:
            out_q.put(("__error__", str(e)))
        finally:
            try:
                done_evt.set()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# === Progress Smoothing & Stage Mapping Helper (PCT MODE, 정확도 개선) ===


# [ADDED] 문서별 base code 아티팩트를 백엔드에서 재수화
def _rehydrate_basecode_from_api(doc_id: int) -> bool:
    try:
        r = requests.get(f"{FASTAPI_URL}/documents/{doc_id}/basecode", timeout=10)
        if not r.ok:
            return False
        data = r.json()
        if not data.get("exists"):
            return False
        # 세션 상태 업데이트
        st.session_state["basecode_py_path"] = data.get("py_path")
        st.session_state["basecode_source"] = data.get("source")
        st.session_state["basecode_summary"] = data.get("summary")
        st.session_state["basecode_error"] = None
        return True
    except Exception:
        return False


# [ADDED] Base code(템플릿 codegen 결과) 섹션 렌더링 유틸
def _show_basecode_section(doc_or_state: dict):
    """
    Base code 아티팩트(파일 경로/소스/요약/스펙)를 Streamlit에 표시한다.
    - 백엔드가 새 필드를 리턴하지 않더라도, 기존 base_code 텍스트만 표시하는 UI는 유지.
    - 입력은 documents API 응답(doc_info) 또는 세션 상태 dict여도 동작하도록 키 탐색.
    """
    st.subheader("🧱 Base Code (논문 구조 재현)")
    st.warning(
        "알림: LLM이 생성한 베이스 코드는 참고용 초안입니다. 사용 전 검토·보완이 필요하며, "
        "실행/학습 과정에서 컴파일 또는 실행 오류가 발생할 수 있습니다."
    )

    # 우선순위: doc_info의 키 → 세션 상태의 키
    def _get(k, default=None):
        return (
            doc_or_state.get(k) if isinstance(doc_or_state, dict) else None
        ) or st.session_state.get(k, default)

    basecode_error = _get("basecode_error")
    py_path = _get("basecode_py_path")
    source = _get("basecode_source")
    summary = _get("basecode_summary")
    spec = _get("spec")
    spec_warnings = _get("spec_warnings") or []

    # 에러가 있으면 먼저 노출
    if basecode_error:
        st.error(f"Base code 생성 실패: {basecode_error}")

    # 새 아티팩트가 없을 경우에도 기존 base_code 텍스트가 있으면 그대로 노출(호환)
    legacy_code = _get("base_code") or _get("basecode") or _get("base-code")

    # 아티팩트가 하나도 없고, 레거시 코드도 없으면 안내
    if not (py_path or source or summary or legacy_code):
        st.info("아직 base code 아티팩트가 없습니다. 문서 분석 후 자동 생성됩니다.")
        return

    # 다운로드 버튼 (파일 경로가 있을 때만)
    if py_path and Path(py_path).exists():
        with open(py_path, "rb") as f:
            st.download_button(
                label="💾 Base code 다운로드(.py)",
                data=f.read(),
                file_name=Path(py_path).name,
                mime="text/x-python",
            )
    elif py_path:
        # 경로가 있지만 실제 파일이 없을 때
        st.warning("생성 파일 경로가 존재하지 않습니다.")

    # 모델 요약
    # if summary:
    #     st.caption("Model Summary (Keras)")
    #     st.code(summary, language="text")

    # 생성된 소스 코드
    if source:
        st.caption("Generated Source (.py)")
        st.code(source, language="python")

    # 레거시 텍스트 기반 base_code (백엔드가 옛 필드만 제공할 때)
    if legacy_code and not source:
        st.caption("Generated Source (.py) [legacy]")
        st.code(legacy_code, language="python")

    # 검증된 스펙/경고 (있을 때만)
    if spec:
        with st.expander("📦 Verified Spec (codegen 입력)"):
            try:
                st.json(spec)
            except Exception:
                st.write(spec)
    if spec_warnings:
        with st.expander("⚠️ Spec Warnings"):
            for w in spec_warnings:
                st.warning(str(w))


load_dotenv()
FASTAPI_URL = "http://localhost:8000"
st.set_page_config(page_title="기술논문 분석 Agent", page_icon="🤖")

# 사용자 정보 로드
try:
    user_info = requests.get(f"{FASTAPI_URL}/users/1").json()
    user_email = user_info["email"]
except:
    user_email = "test@example.com"

# ✅ 세션 상태 초기화
for key, default in {
    "selected_doc_id": None,
    "is_new_analysis": False,
    "qa_list": [],
    "base_code": "",  # ✅ 추가
    # [ADDED] 새 아티팩트 필드 (없으면 생성)
    "basecode_py_path": None,
    "basecode_source": None,
    "basecode_summary": None,
    "spec": None,
    "spec_warnings": [],
    "basecode_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ✅ 히스토리 렌더 함수
def render_qa_history(placeholder, qa_list):
    """QA 히스토리를 placeholder 영역에 렌더링"""
    with placeholder.container():
        st.subheader("💬 질문/응답 히스토리")
        if qa_list:
            for qa in qa_list:
                st.markdown(f"**Q:** {qa['question']}")
                st.markdown(f"**A:** {qa['answer']}")
                st.markdown(
                    f"<small>{qa.get('created_at', '')}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
        else:
            st.info("❗ 질문 히스토리가 없습니다.")


# ----------------- 🔹 사이드바 -------------------
with st.sidebar:
    st.markdown(f"👤 **사용자:** {user_email}")
    st.markdown("### 📁 문서 선택")

    try:
        resp = requests.get(f"{FASTAPI_URL}/documents")
        resp.raise_for_status()
        doc_list = resp.json()
    except Exception as e:
        st.error(f"문서 목록 조회 실패: {e}")
        doc_list = []

    NEW_LABEL = "📤 새 논문 분석 시작"
    options = [NEW_LABEL] + [f"{d['filename']} (ID: {d['id']})" for d in doc_list]

    # ✅ 옵션 변경 감지 → 위젯 상태 초기화
    OPT_SNAPSHOT_KEY = "_doc_options_snapshot"
    if st.session_state.get(OPT_SNAPSHOT_KEY) != options:
        st.session_state.pop("doc_selectbox", None)  # 이전 선택값 제거
        st.session_state[OPT_SNAPSHOT_KEY] = options

    def _current_index():
        sel_id = st.session_state.get("selected_doc_id")
        if sel_id is None:
            return 0
        for i, d in enumerate(doc_list, start=1):
            if d["id"] == sel_id:
                return i
        return 0

    # ✅ 인덱스 범위 안전 가드
    idx = _current_index()
    if len(options) == 0:
        options = [NEW_LABEL]
    idx = max(0, min(idx, len(options) - 1))

    # selected = st.selectbox(
    #     "문서를 선택하세요", options, index=_current_index(), key="doc_selectbox"
    # )
    selected = st.selectbox(
        "문서를 선택하세요", options, index=idx, key="doc_selectbox"
    )

    prev_doc_id = st.session_state.get("selected_doc_id")
    prev_is_new = st.session_state.get("is_new_analysis", False)

    if selected == NEW_LABEL:
        st.session_state["selected_doc_id"] = None
        st.session_state["is_new_analysis"] = True
        st.session_state["qa_list"] = []
        st.session_state["base_code"] = ""  # ✅ 리셋
        # [ADDED] 새 아티팩트 필드 리셋
        for k in [
            "basecode_py_path",
            "basecode_source",
            "basecode_summary",
            "spec",
            "spec_warnings",
            "basecode_error",
        ]:
            st.session_state[k] = None if k != "spec_warnings" else []
        st.session_state.pop("qa_placeholder", None)

        if prev_doc_id is not None or prev_is_new is False:
            st.rerun()
    else:

        st.markdown("---")
        if st.button("🗑️ 선택한 문서 삭제"):
            try:
                del_id = int(selected.split("ID: ")[-1].rstrip(")"))
                res = requests.delete(f"{FASTAPI_URL}/documents/{del_id}")
                res.raise_for_status()

                st.success("✅ 문서가 삭제되었습니다.")
                st.session_state["selected_doc_id"] = None
                st.session_state["is_new_analysis"] = True
                st.session_state["qa_list"] = []
                # st.session_state["base_code"] = ""         # ← 선택
                st.session_state.pop("doc_selectbox", None)  # ← 위젯 리셋
                st.rerun()
            except Exception as e:
                st.error(f"❌ 삭제 실패: {e}")

        doc_id = int(selected.split("ID: ")[-1].rstrip(")"))
        if prev_doc_id != doc_id:
            st.session_state["selected_doc_id"] = doc_id
            st.session_state["is_new_analysis"] = False
            st.session_state["base_code"] = (
                ""  # ✅ 다른 문서로 갈 때 이전 코드 잔상 제거
            )
            # [ADDED] 아티팩트 필드도 함께 리셋 (문서 전환 시 잔상 제거)
            for k in [
                "basecode_py_path",
                "basecode_source",
                "basecode_summary",
                "spec",
                "spec_warnings",
                "basecode_error",
            ]:
                st.session_state[k] = None if k != "spec_warnings" else []
            st.session_state.pop(
                "qa_placeholder", None
            )  # [ADDED] 낡은 placeholder 키 제거

            st.session_state.pop("doc_selectbox", None)  # ← 옵션/선택 재정렬 시 안전

            try:
                r = requests.get(f"{FASTAPI_URL}/qa/{doc_id}")
                r.raise_for_status()
                st.session_state["qa_list"] = r.json()
            except:
                st.session_state["qa_list"] = []
            st.rerun()

# -------------------- 🔹 Main View --------------------
st.title("📄 AI 기술논문 Researh Agent")

# ✅ 문서 업로드 화면 (새 문서 분석)
if st.session_state["selected_doc_id"] is None and st.session_state["is_new_analysis"]:
    uploaded_file = st.file_uploader(
        "💾 논문자료를 업로드하세요. (Only PDF)", type=["pdf"]
    )
    if uploaded_file:
        # === Progress-driven pipeline (threaded, NO SPINNER) ===
        steps = ["파일 업로드", "서버 분석 진행 중", "결과 반영", "화면 갱신"]

        # ✅ 텍스트 체크리스트와 HTML 스피너는 분리된 placeholder 사용
        steps_text = st.empty()  # 텍스트 마커(기존 _render_steps 용)
        steps_box = st.empty()  # HTML 스피너 렌더용
        progress = st.progress(0, text="시작합니다...")

        # Step 1
        # steps_text.markdown(_render_steps(0, steps))
        progress.progress(10, text="📤 파일 업로드 준비...")
        time.sleep(0.35)

        if uploaded_file is None:
            st.warning("업로드된 파일이 없습니다.")
            st.stop()

        _file_bytes = uploaded_file.getvalue()
        _filename = getattr(uploaded_file, "name", "uploaded.pdf")

        _an_done = threading.Event()
        _an_q = queue.Queue(maxsize=1)
        _ = _start_analysis_job_bytes(
            _file_bytes, _filename, FASTAPI_URL, _an_done, _an_q
        )

        progress.progress(25, text="📤 업로드/요청 전송 완료")
        time.sleep(0.45)

        # Step 2
        # steps_text.markdown(_render_steps(1, steps))

        # ✅ 스무딩: 업로드 시작 시 만든 '_an_done' 이벤트를 그대로 사용
        pct = _smooth_wait_with_substeps(
            done_evt=_an_done,
            steps_box=steps_box,
            progress=progress,
            min_pct=30,
            max_pct=99,
            max_secs=150,
        )

        if _an_done.is_set():
            try:
                _out = _an_q.get(timeout=2.0)
            except Exception:
                _out = ("__error__", "결과 큐가 비어 있습니다(타이밍 문제).")
            if isinstance(_out, tuple) and _out and _out[0] == "__error__":
                progress.progress(pct, text="⛔ 서버 분석 실패")
                st.error(f"분석 실패: {_out[1]}")
                st.stop()
            else:
                progress.progress(99, text="✅ 서버 분석 완료")
                result = _out
        else:
            progress.progress(pct, text="⏳ 서버 응답 지연 (타임아웃)")
            st.warning("서버 분석 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.")
            st.stop()

        # Step 3
        # steps_text.markdown(_render_steps(2, steps))
        progress.progress(96, text="📦 결과 반영 중...")

        for k in [
            "document_id",
            "title",
            "meta",
            "preview",
            "embedding_stats",
            "suggested_questions",
            "basecode_py_path",
            "basecode_source",
            "basecode_summary",
            "spec",
            "spec_warnings",
            "basecode_error",
        ]:
            if k in result:
                st.session_state[k] = result[k]

        st.session_state["selected_doc_id"] = result.get("document_id")
        st.session_state["is_new_analysis"] = False
        st.session_state["qa_list"] = []
        st.session_state.pop("doc_selectbox", None)

        # Step 4
        # steps_text.markdown(_render_steps(3, steps))
        progress.progress(100, text="🖥️ 렌더링 완료")
        st.success("✅ 문서 분석 완료!")
        st.rerun()

# ✅ 기존 문서 조회 화면
elif st.session_state["selected_doc_id"] is not None:
    doc_id = st.session_state["selected_doc_id"]
    try:
        docs = requests.get(f"{FASTAPI_URL}/documents").json()
        doc_info = next((doc for doc in docs if doc["id"] == doc_id), None)

        with st.sidebar.expander("🔎  Backend response (/documents current doc)"):
            st.write("doc keys:", list(doc_info.keys()) if doc_info else None)
            st.json(doc_info)

        if doc_info:
            st.subheader("🧠 기술 도메인")
            st.markdown(f"{doc_info['domain']}")

            st.subheader("📝 문서 요약")
            st.write(doc_info["summary"])

            # -------------------- [ADDED] 재수화 선행 --------------------
            # doc_info에 아티팩트가 없고, 세션에도 없으면 먼저 백엔드에서 재수화
            need_rehydrate = not (
                (
                    doc_info
                    and any(
                        k in doc_info and doc_info[k]
                        for k in ("basecode_source", "basecode_py_path")
                    )
                )
                or st.session_state.get("basecode_source")
                or st.session_state.get("basecode_py_path")
            )
            if need_rehydrate:
                with contextlib.nullcontext():
                    try:
                        _rehydrate_basecode_from_api(doc_id)
                    except Exception as _e:
                        # 재수화 실패해도 아래 섹션에서 메시지로 안내됨
                        pass
            # ------------------------------------------------------------

            # Base code 섹션 (재수화 이후 렌더)
            _show_basecode_section(doc_info)  # [MODIFIED] ← 위치 이동

            # QA 히스토리 로딩
            qa_list = st.session_state.get("qa_list", [])
            if not qa_list:
                try:
                    r = requests.get(f"{FASTAPI_URL}/qa/{doc_id}")
                    if r.ok:
                        qa_list = r.json()
                        st.session_state["qa_list"] = qa_list
                except:
                    pass

            # [MODIFIED] placeholder 캐싱 없이 매번 새 컨테이너로 렌더
            placeholder = st.empty()
            render_qa_history(placeholder, qa_list)

        else:
            st.error("❌ 문서 정보를 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"문서 조회 실패: {e}")


# ✅ 질문 입력창 (하단 고정)
user_question = st.chat_input("질문을 입력하세요.")
doc_id = st.session_state.get("selected_doc_id")

if user_question and doc_id is not None:
    with st.spinner("⏳ 답변 생성 중..."):
        try:
            r = requests.post(
                f"{FASTAPI_URL}/qa/ask_existing",
                json={"document_id": doc_id, "question": user_question},
            )
            r.raise_for_status()
            ans = r.json().get("answer", "")
            KST = timezone(timedelta(hours=9))
            created_at_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")

            st.session_state["qa_list"].append(
                {
                    "question": user_question,
                    "answer": ans,
                    "created_at": created_at_str,
                }
            )

            # 🔁 전체 다시 렌더링!
            st.rerun()

        except Exception as e:
            st.error(f"❌ 질문 실패: {e}")
