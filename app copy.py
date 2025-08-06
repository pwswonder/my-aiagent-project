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

# # st.sidebar.markdown(f"👤 현재 사용자: `{user_email}`")
# with st.sidebar:
#     st.markdown("## 📂 업로드된 문서")

#     try:
#         response = requests.get(f"{FASTAPI_URL}/documents")
#         response.raise_for_status()
#         doc_list = response.json()
#     except Exception as e:
#         st.error(f"문서 목록 조회 실패: {e}")
#         doc_list = []


#     if doc_list:
#         filenames = [f"{doc['filename']} (ID: {doc['id']})" for doc in doc_list]
#         selected_doc = st.selectbox("문서를 선택하세요", filenames)

#         selected_doc_id = int(selected_doc.split("ID: ")[-1].strip(")"))
#         doc_info = next((d for d in doc_list if d["id"] == selected_doc_id), None)

#         if doc_info:
#             st.markdown(f"**🧠 기술 도메인:** `{doc_info.get('domain', 'N/A')}`")
#             st.markdown("**📝 문서 요약:**")
#             st.write(doc_info.get("summary", "요약 정보 없음"))

#             st.markdown("**💬 질문/응답 히스토리**")
#             try:
#                 qa_res = requests.get(f"{FASTAPI_URL}/qa/{selected_doc_id}")
#                 qa_res.raise_for_status()
#                 qa_list = qa_res.json()

#                 if qa_list:
#                     for qa in qa_list:
#                         st.markdown(f"**Q:** {qa['question']}")
#                         st.markdown(f"**A:** {qa['answer']}")
#                         st.markdown(
#                             f"<small>{qa['created_at']}</small>", unsafe_allow_html=True
#                         )
#                         st.markdown("---")
#                 else:
#                     st.info("❗ 히스토리 없음.")
#             except Exception as e:
#                 st.error(f"히스토리 조회 실패: {e}")

#             # ✅ 기존 문서에 대한 추가 질문
#             st.markdown("**➕ 새로운 질문하기**")
#             new_question = st.text_input("질문을 입력하세요", key="extra_question")

#             if st.button("질문 제출", key="submit_extra_question"):
#                 if new_question.strip() == "":
#                     st.warning("질문을 입력해주세요.")
#                 else:
#                     try:
#                         with st.spinner("질문 응답 생성 중..."):
#                             res = requests.post(
#                                 f"{FASTAPI_URL}/qa/ask_existing",
#                                 json={
#                                     "document_id": selected_doc_id,
#                                     "question": new_question,
#                                 },
#                             )
#                             res.raise_for_status()
#                             result = res.json()
#                             st.success("✅ 질문 처리 완료!")
#                             st.markdown("**📤 답변:**")
#                             st.write(result["answer"])
#                             st.rerun()  # 질문/답변 히스토리 갱신
#                     except Exception as e:
#                         st.error(f"질문 처리 실패: {e}")

#             # ✅ 삭제 버튼
#             if st.button("🗑️ 선택한 문서 삭제"):
#                 try:
#                     del_res = requests.delete(
#                         f"{FASTAPI_URL}/documents/{selected_doc_id}"
#                     )
#                     del_res.raise_for_status()
#                     st.success("✅ 문서가 성공적으로 삭제되었습니다.")
#                     st.rerun()  # 자동 새로고침
#                 except Exception as e:
#                     st.error(f"❌ 삭제 실패: {e}")

#     else:
#         st.info("업로드된 문서가 없습니다.")


st.title("📄 AI 기술논문 Agent")


uploaded_file = st.file_uploader(
    "💾 논문자료를 업로드하세요. (Only PDF파일)", type=["pdf"]
)
# 2. 질문 입력
# user_question = st.text_input("⌨️ 질문을 입력하세요 (예: 이 논문의 핵심기술이 뭐야?)")


# 3. 사용자 입력 모두 완료된 경우 → FastAPI 요청
# if uploaded_file and user_question:
#     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
#     data = {"question": user_question}

#     with st.spinner("⏳ 문서 분석 및 질문 응답 중입니다..."):
#         try:
#             # FastAPI로 POST 요청 보내기
#             response = requests.post(
#                 f"{FASTAPI_URL}/documents/upload", files=files, data=data
#             )
#             response.raise_for_status()
#             result = response.json()

#             # # 🔍 로그 확인용 출력
#             # st.write(result)
#             # st.code(result, language="json")

#             # ✅ 중복 문서 여부 확인
#             if result.get("message") == "File already uploaded.":
#                 st.info("⚠️ 이미 업로드된 문서입니다. 기존 분석 결과를 불러왔습니다.")
#             else:
#                 st.success("✅ 분석 완료!")

#             # 결과 출력 (중복/신규 모두 공통)
#             if "answer" in result:
#                 st.subheader("📤 질문에 대한 응답")
#                 st.write(result["answer"])

#             if "summary" in result:
#                 st.subheader("📝 문서 요약")
#                 st.write(result["summary"])

#             if "domain" in result:
#                 st.subheader("🧠 기술 도메인")
#                 st.markdown(f"`{result['domain']}`")

#         except requests.exceptions.RequestException as e:
#             st.error(f"❌ FastAPI 요청 실패: {e}")


# 문서만 업로드 버튼
if uploaded_file and st.button("📂 문서 분석만 하기"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

    with st.spinner("⏳ 문서 분석 중입니다..."):
        try:
            response = requests.post(
                f"{FASTAPI_URL}/documents/analyze_only", files=files
            )
            response.raise_for_status()
            result = response.json()

            st.success("✅ 문서 분석 완료!")
            # st.write(result)
            if "domain" in result:
                st.subheader("🧠 기술 도메인")
                st.markdown(f"`{result['domain']}`")
            if "summary" in result:
                st.subheader("📝 문서 요약")
                st.write(result["summary"])



        except Exception as e:
            st.error(f"❌ 분석 실패: {e}")

    user_question = st.text_input("⌨️ 질문을 입력하세요 (예: 이 논문의 핵심기술이 뭐야?)")
    data = {"question": user_question}

    with st.spinner("⏳ 문서 분석 및 질문 응답 중입니다..."):
        try:
            # FastAPI로 POST 요청 보내기
            response = requests.post(
                f"{FASTAPI_URL}/documents/upload", files=files, data=data
            )
            response.raise_for_status()
            result = response.json()

                        # 결과 출력 (중복/신규 모두 공통)
            if "answer" in result:
                st.subheader("📤 질문에 대한 응답")
                st.write(result["answer"])

        except requests.exceptions.RequestException as e:
            st.error(f"❌ FastAPI 요청 실패: {e}")
