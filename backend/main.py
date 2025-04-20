from fastapi import FastAPI
import joblib
import os
import numpy as np
from pydantic import BaseModel
from backend.database import engine, Base
from backend.routes.audio import router as audio_router
from backend.routes.recognition import router as recognition_router

# Khởi tạo FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Chỉ định frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo bảng trong cơ sở dữ liệu (nếu chưa có)
Base.metadata.create_all(bind=engine)

# Đăng ký router
app.include_router(audio_router)

# Đăng ký router nhận diện
app.include_router(recognition_router)

# Chạy server (nếu chạy file này trực tiếp)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
