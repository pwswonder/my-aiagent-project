import streamlit as st
from services.file_reader import file_reader
from services.graph_builder import build_graph
from langsmith import traceable
from dotenv import load_dotenv
import requests


load_dotenv()

st.set_page_config(page_title="기술논문 분석 Agent", page_icon="🤖")
st.title("📄 AI 기술논문 요약 및 도메인 분석 Agent")


# user_id = 1

# # 문서 목록 조회
# doc_response = requests.get(
#     "http://localhost:8000/qa/history/docs", params={"user_id": user_id}
# )


# # 1. 파일 업로드 및 질문 입력
# uploaded_file = st.file_uploader("🔼 PDF 파일을 업로드하세요", type=["pdf"])
# question = st.text_input("💬 질문을 입력하세요")

# # 2. 버튼 누르면 FastAPI에 POST 요청
# if uploaded_file and question:
#     if st.button("📡 질문 전송"):
#         with st.spinner("AI가 응답을 생성 중입니다..."):
#             try:
#                 # FastAPI로 파일 + 질문 전송
#                 response = requests.post(
#                     "http://localhost:8000/qa/submit",
#                     files={"file": uploaded_file},
#                     data={"question": question},
#                 )
#                 if response.status_code == 200:
#                     answer = response.json().get("answer")
#                     st.success(f"🤖 AI 응답: {answer}")
#                 else:
#                     st.error(f"❌ 서버 오류 (status: {response.status_code})")
#             except Exception as e:
#                 st.error(f"❌ 요청 실패: {e}")

# # 사용자 ID는 하드코딩 또는 세션 기반으로 처리 (예: 1번 사용자)


# st.subheader("📚 이전 질문/답변 히스토리 보기")

# # 문서 목록 조회 (✅ URL 수정)
# if doc_response.status_code == 200:
#     docs = doc_response.json()
#     if docs:
#         doc_options = {f"{doc['title']} (ID: {doc['id']})": doc["id"] for doc in docs}
#         selected_doc = st.selectbox(
#             "문서를 선택하세요", options=list(doc_options.keys())
#         )

#         if selected_doc:
#             doc_id = doc_options[selected_doc]
#             hist_response = requests.get(f"http://localhost:8000/qa/history/{doc_id}")
#             if hist_response.status_code == 200:
#                 histories = hist_response.json()
#                 for item in histories:
#                     with st.expander(f"❓ {item['user_input']}"):
#                         st.markdown(f"🧠 {item['ai_answer']}")
#             else:
#                 st.error("히스토리를 불러오지 못했습니다.")
#     else:
#         st.info("업로드된 문서가 없습니다.")
# else:
#     st.error("문서 목록을 불러오지 못했습니다.")


uploaded_file = st.file_uploader(
    "💾 논문자료를 업로드하세요. (Only PDF파일)", type=["pdf"]
)

if uploaded_file:
    temp_file_path = f"/tmp/{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    state = file_reader({"file": temp_file_path})

    # 디비깅
    # st.subheader("📃 추출된 문서 내용 (앞부분)")
    # st.write(state["raw_text"][:1000])

    user_input = st.text_input("⌨️ 질문을 입력하세요 (예: 이 논문의 핵심기술이 뭐야?)")

    if user_input:
        state["user_input"] = user_input

        graph = build_graph()
        st.info("⏳ AI Agent 열일중...")

        result = graph.invoke(state)

        # st.success("결과 도출 완료!!")
        st.write("🔍 retriever 존재여부: ", "retriever" in result)

        if "answer" in result:
            st.subheader("📤 답변")
            st.write(result["answer"])
        elif "summary" in result:
            st.subheader("📤 요약")
            st.write(result["summary"])
        elif "domain" in result:
            st.subheader("📤 분류")
            st.write(result["domain"])
