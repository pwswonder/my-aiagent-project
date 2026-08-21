from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API = os.getenv("PAPER_AGENT_API_URL", "http://localhost:8000").rstrip("/")
POLL_SECONDS = 2

st.set_page_config(page_title="AI Paper Agent", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child
    div[data-testid="stButton"] {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child button {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 3.25rem;
        padding: 0;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📄 AI Paper Agent")
st.caption("PDF를 업로드하면 논문 분석, PyTorch 코드 생성과 검증까지 자동으로 진행합니다.")

st.session_state.setdefault("selected_document_id", None)
st.session_state.setdefault("uploader_nonce", 0)


def request(method: str, path: str, **kwargs: Any) -> requests.Response | None:
    timeout = kwargs.pop("timeout", 30)
    try:
        response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        st.error(f"서버에 연결할 수 없습니다: {exc}")
        return None
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"요청을 처리하지 못했습니다: {detail}")
        return None
    return response


def status_label(document: dict[str, Any]) -> str:
    status = document.get("analysis_status") or document.get("status") or "unknown"
    labels = {
        "queued": "대기 중",
        "running": "분석 중",
        "completed": "분석 완료",
        "failed": "실패",
        "needs_review": "검토 필요",
        "spec_approved": "모델 구조 확인",
        "generating": "코드 생성 중",
    }
    return labels.get(status, status)


history_response = request("GET", "/api/v2/documents", timeout=20)
history = history_response.json() if history_response is not None else []

with st.sidebar:
    st.header("논문 히스토리")
    uploaded = st.file_uploader(
        "새 논문 업로드",
        type=["pdf"],
        key=f"paper_upload_{st.session_state.uploader_nonce}",
        help="PDF를 선택하면 별도 버튼 없이 분석이 시작됩니다.",
    )
    if uploaded is not None:
        with st.spinner("업로드하고 분석 작업을 시작하는 중입니다..."):
            response = request(
                "POST",
                "/api/v2/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=90,
            )
        if response is not None:
            payload = response.json()
            st.session_state.selected_document_id = payload["document_id"]
            st.session_state.uploader_nonce += 1
            st.rerun()

    st.divider()
    if not history:
        st.caption("아직 분석한 논문이 없습니다.")
    for document in history:
        title = document.get("title") or document["filename"]
        selected = document["id"] == st.session_state.selected_document_id
        label = f"{'●' if selected else '○'} {title}\n\n{status_label(document)}"
        item_column, delete_column = st.columns([0.84, 0.16], vertical_alignment="center")
        with item_column:
            if st.button(label, key=f"history_{document['id']}", width="stretch"):
                st.session_state.selected_document_id = document["id"]
                st.rerun()
        with delete_column:
            if st.button(
                "🗑️",
                key=f"delete_{document['id']}",
                help=f"{title}을(를) 히스토리에서 삭제",
                width="stretch",
            ):
                deleted = request("DELETE", f"/api/v2/documents/{document['id']}", timeout=30)
                if deleted is not None:
                    if selected:
                        st.session_state.selected_document_id = None
                    st.rerun()

if st.session_state.selected_document_id is None and history:
    st.session_state.selected_document_id = history[0]["id"]
    st.rerun()

if st.session_state.selected_document_id is None:
    st.info("왼쪽에서 PDF 논문을 업로드하면 분석이 자동으로 시작됩니다.")
    st.stop()

document_id = st.session_state.selected_document_id
workspace_response = request("GET", f"/api/v2/documents/{document_id}/workspace", timeout=30)
if workspace_response is None:
    st.stop()
workspace = workspace_response.json()
analysis_run = workspace.get("analysis_run")

st.header(workspace.get("title") or workspace["filename"])
st.caption(workspace["filename"])

if analysis_run and analysis_run["status"] in {"queued", "running"}:
    progress = analysis_run.get("progress", 0)
    st.progress(progress / 100, text=f"{analysis_run.get('stage', '분석 중')} · {progress}%")
    st.info("분석이 끝나면 이 화면에 요약과 모델 구조가 자동으로 표시됩니다.")
    with st.expander("분석 진행 내역"):
        for event in analysis_run.get("event_log", []):
            st.write(f"{event.get('progress', 0)}% · {event.get('message', event.get('stage', ''))}")
    time.sleep(POLL_SECONDS)
    st.rerun()

if analysis_run and analysis_run["status"] == "failed":
    st.error(f"분석에 실패했습니다: {analysis_run.get('error') or '원인을 확인할 수 없습니다.'}")
    st.stop()

st.subheader("논문 요약")
if workspace.get("summary"):
    st.write(workspace["summary"])
else:
    st.info("요약을 생성할 근거가 아직 준비되지 않았습니다.")

st.divider()
st.subheader("모델 구조와 코드")
spec_payload = workspace.get("spec")
generation = workspace.get("generation")
generation_run = workspace.get("generation_run")

if not spec_payload:
    st.info("모델 구조를 분석하고 있습니다.")
else:
    spec = spec_payload["spec"]
    nodes = spec.get("nodes", [])
    unresolved = spec.get("unresolved", [])
    blocking = [item for item in unresolved if item.get("blocking", True)]

    left, middle, right = st.columns(3)
    left.metric("모델 블록", len(nodes))
    middle.metric("근거", len(spec.get("evidence", [])))
    right.metric("검토 항목", len(unresolved))

    if nodes:
        with st.expander("분석된 모델 구조와 근거 보기"):
            st.dataframe(
                [
                    {
                        "블록": node["id"],
                        "연산": node["op"],
                        "입력": ", ".join(node["inputs"]),
                        "출력": node["output"],
                        "confidence": node.get("confidence", 1.0),
                    }
                    for node in nodes
                ],
                width="stretch",
                hide_index=True,
            )
            if unresolved:
                st.caption("논문에 명시되지 않은 세부값은 생성 provenance에 가정으로 기록했습니다.")
                for item in unresolved:
                    st.markdown(f"- **{item['field']}**: {item['question']}")
    else:
        st.error("논문에서 구현 가능한 모델 구조를 추출하지 못했습니다.")
        if st.button("논문 다시 분석", type="primary"):
            response = request("POST", f"/api/v2/documents/{document_id}/reanalyze", timeout=30)
            if response is not None:
                st.rerun()

    # Compatibility for documents analyzed before the automatic pipeline was
    # introduced. New analyses already queue generation in the worker.
    if nodes and spec_payload["status"] != "approved" and generation is None:
        with st.spinner("분석된 구조로 코드 생성을 자동 시작합니다..."):
            approved = request(
                "POST",
                f"/api/v2/documents/{document_id}/spec/{spec_payload['version']}/approve",
                json={"accept_blocking_as_assumptions": bool(blocking)},
                timeout=30,
            )
            if approved is not None:
                created = request(
                    "POST",
                    f"/api/v2/documents/{document_id}/generations",
                    timeout=30,
                )
                if created is not None:
                    st.rerun()
    elif nodes and spec_payload["status"] == "approved" and generation is None:
        with st.spinner("승인된 IR로 코드 생성 작업을 시작합니다..."):
            created = request("POST", f"/api/v2/documents/{document_id}/generations", timeout=30)
        if created is not None:
            st.rerun()

if generation:
    status = generation["status"]
    if status in {"queued", "running"} or (generation_run and generation_run["status"] in {"queued", "running"}):
        progress = generation_run.get("progress", 0) if generation_run else 0
        stage = generation_run.get("stage", "코드 생성 중") if generation_run else "코드 생성 중"
        st.progress(progress / 100, text=f"{stage} · {progress}%")
        time.sleep(POLL_SECONDS)
        st.rerun()
    else:
        preview = request("GET", f"/api/v2/generations/{generation['id']}/preview", timeout=30)
        if preview is not None:
            model_source = preview.json().get("files", {}).get("model.py")
            if model_source:
                st.markdown("#### 구현된 `model.py`")
                st.caption("코드 영역 안에서 스크롤할 수 있습니다.")
                st.code(
                    model_source,
                    language="python",
                    line_numbers=True,
                    wrap_lines=False,
                    height=600,
                )

    if status == "completed":
        artifact_id = generation.get("artifact_id")
        if artifact_id:
            st.link_button(
                "코드 패키지 다운로드",
                f"{API}/api/v2/artifacts/{artifact_id}/download",
                type="primary",
            )

st.divider()
st.subheader("논문에 질문하기")
qa_history = workspace.get("qa_history", [])
if not qa_history:
    st.caption("아직 질문 내역이 없습니다. 아래 입력창에서 논문에 대해 질문해 보세요.")
for turn in qa_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("answer"):
            st.write(turn["answer"])
            citations = turn.get("citations", [])
            if citations:
                with st.expander("논문 근거"):
                    for citation in citations:
                        st.markdown(f"- p.{citation['page']} · {citation['section']}: {citation['evidence']}")
        else:
            st.warning("논문에서 답변 근거를 찾지 못했습니다.")

question = st.chat_input("이 논문의 방법, 실험, 한계에 대해 질문하세요")
if question:
    with st.spinner("논문 근거를 검색하고 답변을 생성하는 중입니다..."):
        response = request(
            "POST",
            f"/api/v2/documents/{document_id}/questions",
            json={"question": question},
            timeout=120,
        )
    if response is not None:
        st.rerun()
