import streamlit as st
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

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
    st.markdown(f"👤 **사용자:** `{user_email}`")
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

    def _current_index():
        sel_id = st.session_state.get("selected_doc_id")
        if sel_id is None:
            return 0
        for i, d in enumerate(doc_list, start=1):
            if d["id"] == sel_id:
                return i
        return 0

    selected = st.selectbox(
        "문서를 선택하세요", options, index=_current_index(), key="doc_selectbox"
    )

    prev_doc_id = st.session_state.get("selected_doc_id")
    prev_is_new = st.session_state.get("is_new_analysis", False)

    if selected == NEW_LABEL:
        st.session_state["selected_doc_id"] = None
        st.session_state["is_new_analysis"] = True
        st.session_state["qa_list"] = []
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
                st.rerun()
            except Exception as e:
                st.error(f"❌ 삭제 실패: {e}")

        doc_id = int(selected.split("ID: ")[-1].rstrip(")"))
        if prev_doc_id != doc_id:
            st.session_state["selected_doc_id"] = doc_id
            st.session_state["is_new_analysis"] = False
            try:
                r = requests.get(f"{FASTAPI_URL}/qa/{doc_id}")
                r.raise_for_status()
                st.session_state["qa_list"] = r.json()
            except:
                st.session_state["qa_list"] = []
            st.rerun()

# -------------------- 🔹 Main View --------------------
st.title("📄 AI 기술논문 Agent")

# ✅ 문서 업로드 화면 (새 문서 분석)
if st.session_state["selected_doc_id"] is None and st.session_state["is_new_analysis"]:
    uploaded_file = st.file_uploader(
        "💾 논문자료를 업로드하세요. (Only PDF)", type=["pdf"]
    )
    if uploaded_file:
        with st.spinner("📄 문서 분석 중입니다..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                response = requests.post(
                    f"{FASTAPI_URL}/documents/analyze_only", files=files
                )
                response.raise_for_status()
                result = response.json()

                # 🔁 업로드 성공 후 상태 갱신 & 상세 화면으로 전환
                st.session_state["selected_doc_id"] = result["document_id"]
                st.session_state["is_new_analysis"] = False
                st.session_state["qa_list"] = []
                st.success("✅ 문서 분석 완료!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ 분석 실패: {e}")

# ✅ 기존 문서 조회 화면
elif st.session_state["selected_doc_id"] is not None:
    doc_id = st.session_state["selected_doc_id"]
    try:
        docs = requests.get(f"{FASTAPI_URL}/documents").json()
        doc_info = next((doc for doc in docs if doc["id"] == doc_id), None)

        if doc_info:
            st.subheader("🧠 기술 도메인")
            st.markdown(f"`{doc_info['domain']}`")

            st.subheader("📝 문서 요약")
            st.write(doc_info["summary"])

            # ✅ 히스토리를 그릴 placeholder 생성
            if "qa_placeholder" not in st.session_state:
                st.session_state["qa_placeholder"] = st.empty()

            qa_list = st.session_state.get("qa_list", [])
            if not qa_list:
                try:
                    r = requests.get(f"{FASTAPI_URL}/qa/{doc_id}")
                    if r.ok:
                        qa_list = r.json()
                        st.session_state["qa_list"] = qa_list
                except:
                    pass

            # ✅ 히스토리 렌더링
            render_qa_history(st.session_state["qa_placeholder"], qa_list)

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
