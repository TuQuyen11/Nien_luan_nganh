from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
import os
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
from scipy.io.wavfile import write
from collections import Counter
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import AudioFile, Feature
from datetime import datetime, timezone
from sqlalchemy import desc, or_


router = APIRouter()


# Thư mục lưu file xử lý
PROCESSED_FOLDER = "processed"
FEATURE_FOLDER = "features"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)

# Load scaler, mô hình nhận diện và mapping ID -> Tên
scaler = joblib.load("backend/model/scaler.pkl")
model = joblib.load("backend/model/svm_model.pkl")
label_mapping = joblib.load("backend/model/label_encoder.pkl")

def process_audio(file_path, output_dir, segment_length=4):
    """Chuẩn hóa và cắt nhỏ file âm thanh."""
    audio = AudioSegment.from_file(file_path).set_channels(1)
    mono_temp_path = file_path.replace(".wav", "_mono.wav")
    audio.export(mono_temp_path, format="wav")

    y, sr = librosa.load(mono_temp_path, sr=16000)
    y = librosa.util.normalize(y)

    segment_samples = segment_length * sr
    num_segments = len(y) // segment_samples
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    processed_files = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        output_file = os.path.join(output_dir, f"{base_name}_segment_{i}.wav")
        write(output_file, sr, (segment * 32767).astype(np.int16))
        processed_files.append(output_file)

    os.remove(mono_temp_path)
    return processed_files

def extract_mfcc(file_path, sr=16000, n_mfcc=30, hop_length=256, n_fft=1024, max_frames=200):
    """Trích xuất đặc trưng MFCC."""
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc.flatten()

def recognize_speaker(feature_vector):
    """Nhận diện người nói và ánh xạ ID sang tên."""
    predicted_id = model.predict([feature_vector])[0]
    try:
        predicted_label = label_mapping.inverse_transform([predicted_id])[0]
    except ValueError:
        predicted_label = "Unknown"
    return predicted_label


@router.post("/recognition/")
async def recognition_audio(file_name: str, db: Session = Depends(get_db)):
    # Chuẩn hóa đường dẫn theo hệ điều hành
    normalized_path = os.path.normpath(f"uploads/{file_name}")

    # Truy vấn file với cả hai kiểu đường dẫn
    audio_entry = (
        db.query(AudioFile)
        .filter(or_(
            AudioFile.file_path == normalized_path,
            AudioFile.file_path.like(f"%{file_name}") 
        ))
        .order_by(desc(AudioFile.uploaded_at))
        .first()
    )
    if not audio_entry:
        return JSONResponse(status_code=404, content={"error": "File not found in database"})

    file_path = audio_entry.file_path
    processed_files = process_audio(file_path, PROCESSED_FOLDER)

    predictions = []
    for processed_file in processed_files:
        feature_vector = extract_mfcc(processed_file)
        mfcc_vector_scaled = scaler.transform([feature_vector])[0].tolist()
        predicted_speaker = recognize_speaker(mfcc_vector_scaled)
        
        # Chuyển `feature_vector` sang `bytes`
        feature_vector_bytes = np.array(mfcc_vector_scaled, dtype=np.float32).tobytes()

        # Lưu vào DB
        new_feature = Feature(
            audio_id=audio_entry.audio_id,
            feature_vector=feature_vector_bytes,
            created_at=datetime.now(timezone.utc)
        )
        db.add(new_feature)
        predictions.append(predicted_speaker)

    db.commit()

    # Kiểm tra nếu tất cả dự đoán đều là "Unknown"
    if all(pred == "Unknown" for pred in predictions):
        return JSONResponse(status_code=200, content={
            "message": "Audio processed but speaker could not be identified.",
            "file_name": file_name,
            "predictions": predictions,
            "final_prediction": "Unknown"
        })

    # Cập nhật kết quả nhận diện vào bảng speakers
    final_prediction = Counter(predictions).most_common(1)[0][0]
    audio_entry.result = final_prediction
    db.commit()
    
    return JSONResponse(content={
        "message": "Audio processed and speaker recognized successfully!",
        "file_name": file_name,
        "predictions": predictions,
        "final_prediction": final_prediction
    })

