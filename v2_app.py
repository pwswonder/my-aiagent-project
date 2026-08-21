from __future__ import annotations

import json
import os
import time

import requests
import streamlit as st


API = os.getenv("PAPER_AGENT_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="AI Paper Agent V2", layout="wide")
st.title("AI Paper Agent V2")
st.caption("Evidence-grounded Architecture IR → reviewed PyTorch package → isolated validation")

for key in ("document_id", "run_id", "generation_id", "generation_run_id", "artifact_id"):
    st.session_state.setdefault(key, None)

upload_tab, analysis_tab, spec_tab, generation_tab, qa_tab = st.tabs(
    ["1. Upload", "2. Analysis", "3. Architecture IR", "4. Generate & validate", "5. Paper QA"]
)

with upload_tab:
    paper = st.file_uploader("Upload a technical paper", type=["pdf"])
    if st.button("Start analysis", type="primary", disabled=paper is None):
        response = requests.post(
            f"{API}/api/v2/documents",
            files={"file": (paper.name, paper.getvalue(), "application/pdf")},
            timeout=90,
        )
        if response.ok:
            payload = response.json()
            st.session_state.document_id = payload["document_id"]
            st.session_state.run_id = payload["analysis_run_id"]
            st.success(f"Queued: {payload['analysis_run_id']}")
        else:
            st.error(response.text)

with analysis_tab:
    run_id = st.text_input("Analysis run ID", value=st.session_state.run_id or "")
    if st.button("Refresh analysis") and run_id:
        response = requests.get(f"{API}/api/v2/runs/{run_id}", timeout=20)
        if response.ok:
            payload = response.json()
            st.progress(payload["progress"] / 100, text=f"{payload['stage']} · {payload['status']}")
            st.json(payload["event_log"])
        else:
            st.error(response.text)

with spec_tab:
    document_id = st.text_input(
        "Document ID", value=st.session_state.document_id or "", key="spec_document"
    )
    if st.button("Load latest IR") and document_id:
        response = requests.get(f"{API}/api/v2/documents/{document_id}/spec", timeout=30)
        if response.ok:
            payload = response.json()
            st.session_state.spec_payload = payload
            st.session_state.spec_editor = json.dumps(payload["spec"], ensure_ascii=False, indent=2)
        else:
            st.error(response.text)
    if "spec_editor" in st.session_state:
        edited = st.text_area("Review graph, shapes, evidence and assumptions", key="spec_editor", height=500)
        left, right = st.columns(2)
        if left.button("Save as new version"):
            try:
                payload = json.loads(edited)
                response = requests.patch(
                    f"{API}/api/v2/documents/{document_id}/spec", json=payload, timeout=30
                )
                response.raise_for_status()
                st.session_state.spec_payload = response.json()
                st.success(f"Saved version {response.json()['version']}")
            except (ValueError, requests.RequestException) as exc:
                st.error(str(exc))
        spec_info = st.session_state.get("spec_payload", {})
        if right.button("Approve this version", disabled=not spec_info):
            version = spec_info["version"]
            response = requests.post(
                f"{API}/api/v2/documents/{document_id}/spec/{version}/approve", timeout=30
            )
            if response.ok:
                st.session_state.spec_payload = response.json()
                st.success("Approved. Code generation is now enabled.")
            else:
                st.error(response.text)

with generation_tab:
    document_id = st.text_input(
        "Approved document ID", value=st.session_state.document_id or "", key="generation_document"
    )
    if st.button("Generate PyTorch package", type="primary", disabled=not document_id):
        response = requests.post(f"{API}/api/v2/documents/{document_id}/generations", timeout=30)
        if response.ok:
            payload = response.json()
            st.session_state.generation_id = payload["generation_id"]
            st.session_state.generation_run_id = payload["run_id"]
            st.success("Generation queued")
        else:
            st.error(response.text)
    generation_id = st.text_input("Generation ID", value=st.session_state.generation_id or "")
    if st.button("Refresh generation") and generation_id:
        response = requests.get(f"{API}/api/v2/generations/{generation_id}", timeout=20)
        if response.ok:
            payload = response.json()
            st.json(payload)
            st.session_state.artifact_id = payload.get("artifact_id")
        else:
            st.error(response.text)
    if st.session_state.artifact_id:
        st.link_button(
            "Download validated package",
            f"{API}/api/v2/artifacts/{st.session_state.artifact_id}/download",
        )

with qa_tab:
    document_id = st.text_input(
        "Analyzed document ID", value=st.session_state.document_id or "", key="qa_document"
    )
    question = st.text_area("Question")
    if st.button("Ask with citations", disabled=not (document_id and question.strip())):
        response = requests.post(
            f"{API}/api/v2/documents/{document_id}/questions",
            json={"question": question},
            timeout=90,
        )
        if response.ok:
            payload = response.json()
            if payload["answerability"] == "answerable":
                st.write(payload["answer"])
                st.subheader("Citations")
                for citation in payload["citations"]:
                    st.markdown(
                        f"- p.{citation['page']} · {citation['section']} · `{citation['chunk_id']}`: "
                        f"{citation['evidence']}"
                    )
            else:
                st.warning("The retrieved paper evidence is insufficient. Rephrase the question.")
            with st.expander("Retrieval debug"):
                st.json(payload["retrieval_debug"])
        else:
            st.error(response.text)
