from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()


class EmbedState(TypedDict):
    raw_text: str
    chunks: list
    vectorstore: FAISS
    retriever: any


def embedder(state: EmbedState) -> EmbedState:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(state["raw_text"])

    embedding_model = AzureOpenAIEmbeddings(
        model=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
        openai_api_version="2024-02-01",
        api_key=os.getenv("AOAI_API_KEY"),
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    )

    vectorstore = FAISS.from_texts(chunks, embedding_model)
    retriever = vectorstore.as_retriever()

    return {
        **state,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "retriever": retriever,
    }
