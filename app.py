import streamlit as st
from services.file_reader import file_reader
from services.graph_builder import build_graph
from langsmith import traceable
from dotenv import load_dotenv
import requests


load_dotenv()

# FastAPI 서버 URL
FASTAPI_URL = "http://localhost:8000"


try:
    user_res = requests.get(f"{FASTAPI_URL}/users/1")  # user_id=1 하드코딩
    user_res.raise_for_status()
    user_info = user_res.json()
    user_email = user_info["email"]
except Exception as e:
    user_email = "Test_1"

st.set_page_config(page_title="기술논문 분석 Agent", page_icon="🤖")


# ✅ 사용자 정보 좌측 상단 표시
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(f"👤 **사용자:** `{user_email}`")

st.title("📄 AI 기술논문 Agent")


uploaded_file = st.file_uploader("💾 논문을 업로드하세요 (PDF만 가능)", type=["pdf"])

document_summary = ""
document_domain = ""
document_id = None
answer = ""

# 업로드 즉시 분석 요청
if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    data = {"question": "이 논문의 도메인과 주요 내용을 요약해줘"}  # 최초 질문

    with st.spinner("📊 문서 분석 중입니다..."):
        try:
            res = requests.post(
                f"{FASTAPI_URL}/documents/upload",
                files=files,
                data=data,
            )
            res.raise_for_status()
            result = res.json()

            # 분석 결과 저장
            document_summary = result.get("summary", "")
            document_domain = result.get("domain", "")
            document_id = result.get("document_id", None)
            answer = result.get("answer", "")

            # ✅ 중복 문서
            if result.get("message") == "File already uploaded.":
                st.info("⚠️ 이미 업로드된 문서입니다. 기존 분석 결과를 불러왔습니다.")
            else:
                st.success("✅ 문서 분석 완료!")

        except requests.exceptions.RequestException as e:
            st.error(f"문서 업로드 또는 분석 실패: {e}")


# --------------------- 분석 결과 표시 ---------------------
if document_id:
    st.subheader("🧠 기술 도메인")
    st.markdown(f"`{document_domain}`")

    st.subheader("📝 문서 요약")
    st.write(document_summary)

    # st.subheader("📤 AI 기본 응답")
    # st.write(answer)

    # --------------------- 사용자 추가 질문 ---------------------
    st.subheader("💬 추가 질문하기")
    user_question = st.text_input("질문을 입력하세요")

    if st.button("질문 전송"):
        if user_question.strip() == "":
            st.warning("질문을 입력해주세요.")
        else:
            with st.spinner("AI가 답변을 생성 중입니다..."):
                try:
                    res = requests.post(
                        f"{FASTAPI_URL}/qa/ask_existing",
                        json={"document_id": document_id, "question": user_question},
                    )
                    res.raise_for_status()
                    result = res.json()
                    st.success("✅ 응답 완료")
                    st.markdown("**📤 답변:**")
                    st.write(result["answer"])
                except Exception as e:
                    st.error(f"❌ 질문 처리 실패: {e}")
