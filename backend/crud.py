from sqlalchemy.orm import Session
from backend import models, schemas


# 사용자
def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# 문서
def create_document(db: Session, doc: schemas.DocumentCreate):
    db_doc = models.Document(**doc.dict())
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


# QA 히스토리
def save_qa_history(db: Session, qa: schemas.QACreate):
    db_qa = models.QAHistories(**qa.dict())
    db.add(db_qa)
    db.commit()
    db.refresh(db_qa)
    return db_qa


def get_qa_by_document(db: Session, document_id: int):
    return (
        db.query(models.QAHistory)
        .filter(models.QAHistory.document_id == document_id)
        .all()
    )


def get_document_by_id(db: Session, document_id: int):
    return db.query(models.Document).filter(models.Document.id == document_id).first()
