from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


# 사용자
class UserBase(BaseModel):
    name: str
    email: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    name: Optional[str]
    email: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# 문서
class DocumentBase(BaseModel):
    user_id: int
    title: str
    file_path: str


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    id: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


# QA 히스토리
class QACreate(BaseModel):
    document_id: int
    user_input: str
    ai_answer: str


class QAHistory(BaseModel):
    id: int
    document_id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True


class QAHistoryOut(BaseModel):
    id: int
    user_input: str
    ai_answer: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: int
    user_id: int
    title: str
    file_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ExistingDocQARequest(BaseModel):
    document_id: int
    question: str
