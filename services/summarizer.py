from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O_MINI"),
    openai_api_version="2024-02-01",
    api_key=os.getenv("AOAI_API_KEY"),
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    temperature=0.3,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document assistant that answers questions based on the given technical paper.",
        ),
        (
            "user",
            "문서: 이 논문은 강화학습 기반으로 드론의 비행 제어를 자동화하였습니다.\n질문: 이 논문의 핵심 기술은 무엇인가요?",
        ),
        (
            "assistant",
            "이 논문의 핵심 기술은 강화학습 알고리즘을 비행 제어 시스템에 적용한 점입니다.",
        ),
        (
            "user",
            "문서: 이 논문은 자율주행 차량의 센서 융합 시스템에 대해 다룹니다.\n질문: 이 논문의 핵심 기술은 무엇인가요?",
        ),
        (
            "assistant",
            "이 논문은 다양한 센서 데이터를 통합하여 자율주행 정확도를 높이는 센서 융합 기술이 핵심입니다.",
        ),
        ("user", "{user_input}"),
    ]
)
chain = prompt | llm | StrOutputParser()

summarizer_agent = RunnableLambda(
    lambda state: {
        "summary": chain.invoke(
            {
                "user_input": f"다음은 논문 내용입니다:\n{state['raw_text'][:5000]}\n\n요약해줘"
            }
        )
    }
)


def qa_with_retrieval(state):
    query = state.get("user_input", "")
    retriever = state.get("retriever", None)
    if retriever is None:
        return {
            "answer": "💡 문서를 임베딩하거나 검색할 수 없습니다. retriever가 없습니다."
        }

    docs = retriever.get_relevant_documents(query)
    if not docs:
        return {
            "answer": "💡 질문과 관련된 문서를 찾을 수 없습니다. 논문 내용이 포함된 PDF를 다시 확인해주세요."
        }

    context = "\n".join([doc.page_content for doc in docs])
    user_prompt = f"다음은 관련 논문 내용입니다:\n{context}\n\n사용자 질문:\n{query}"
    return {"answer": chain.invoke({"user_input": user_prompt})}


qa_agent = RunnableLambda(qa_with_retrieval)
