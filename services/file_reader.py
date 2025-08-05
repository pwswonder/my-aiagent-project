from langchain_community.document_loaders import PyMuPDFLoader
from typing import TypedDict

class DocState(TypedDict):
    file: str  
    raw_text: str

def file_reader(state: DocState) -> DocState:
    loader = PyMuPDFLoader(state["file"])
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    return {**state, "raw_text": text}
