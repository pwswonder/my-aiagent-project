from fastapi import APIRouter, UploadFile, File, Form, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
import os, shutil
from backend.database import SessionLocal, get_db
from sqlalchemy.orm import Session
from backend import models, schemas, crud
from typing import List
from services.file_reader import file_reader
from services.graph_builder import build_graph
from langgraph.graph import StateGraph


router = APIRouter()
UPLOAD_FOLDER = "uploaded_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/qa/history/docs", response_model=List[schemas.DocumentOut])
def get_docs_by_user(
    user_id: int = Query(..., description="사용자 ID"), db: Session = Depends(get_db)
):
    return db.query(models.Document).filter(models.Document.user_id == user_id).all()


@router.post("/qa/submit")
async def handle_question(file: UploadFile = File(...), question: str = Form(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ai_response = f"📘 질문 '{question}'에 대한 응답입니다 (파일: {file.filename})"

    db = SessionLocal()
    document = models.Document(user_id=1, title=file.filename, file_path=file_location)
    db.add(document)
    db.commit()
    db.refresh(document)

    qa = models.QAHistories(
        document_id=document.id, user_input=question, ai_answer=ai_response
    )
    db.add(qa)
    db.commit()
    return JSONResponse(content={"answer": ai_response})


@router.get("/qa/history/{document_id}", response_model=List[schemas.QAHistoryOut])
def get_qa_history(document_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.QAHistories)
        .filter(models.QAHistories.document_id == document_id)
        .all()
    )


@router.post("/qa/ask_existing")
def ask_existing_document_question(
    payload: schemas.ExistingDocQARequest, db: Session = Depends(get_db)
):
    try:
        # 1. 문서 조회
        document = crud.get_document_by_id(db, payload.document_id)
        if not document:
            print("[ERROR] 문서를 찾을 수 없습니다.")
            raise HTTPException(status_code=404, detail="Document not found.")

        print(f"[DEBUG] 문서 경로: {document.file_path}")

        # 2. 텍스트 추출
        file_state = file_reader({"file": document.file_path})
        raw_text = file_state["raw_text"]
        print(f"[DEBUG] 추출된 텍스트 길이: {len(raw_text)}")

        # 3. 그래프 실행
        graph = build_graph()
        result = graph.invoke({"raw_text": raw_text, "user_input": payload.question})

        print("[DEBUG] 그래프 결과:", result)

        if "answer" not in result:
            print("[ERROR] 그래프 결과에 'answer' 없음")
            raise HTTPException(status_code=500, detail="답변 생성 실패")

        # 4. DB 저장
        qa_entry = schemas.QACreate(
            document_id=payload.document_id,
            user_input=payload.question,
            ai_answer=result["answer"],
        )
        crud.save_qa_history(db, qa=qa_entry)

        return {"answer": result["answer"]}

    except Exception as e:
        import traceback

        print("🔴 예외 발생:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"에러 발생: {e}")
