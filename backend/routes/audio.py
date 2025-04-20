from fastapi import APIRouter, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
import shutil
import os
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import AudioFile
from datetime import datetime

router = APIRouter()

########### UPLOAD FILE ####################################################
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@router.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    created_at: str = Form(...),
    db: Session = Depends(get_db)  # Dependency để lấy session DB
):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file_path = file_path.replace("\\", "/")

    # Lưu file vào thư mục
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Chuyển `created_at` từ string sang datetime
    created_at_dt = datetime.strptime(created_at, "%d/%m/%Y %H:%M")

    # Lưu vào CSDL bằng SQLAlchemy 
    new_audio = AudioFile(
        file_path=file_path,
        uploaded_at=created_at_dt
    )
    db.add(new_audio)
    db.commit()
    db.refresh(new_audio)

    return JSONResponse(content={"message": "File uploaded successfully!", "file_name": file.filename})
