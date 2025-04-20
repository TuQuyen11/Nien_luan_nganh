from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base
from sqlalchemy.dialects.postgresql import BYTEA

class AudioFile(Base):
    __tablename__ = "speakers"

    audio_id = Column(Integer, primary_key=True, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, nullable=False)
    
    features = relationship("Feature", back_populates="audio")

class Feature(Base):
    __tablename__ = "features"

    feature_id = Column(Integer, primary_key=True, index=True, nullable=False)
    audio_id = Column(Integer, ForeignKey("speakers.audio_id"), nullable=False)
    feature_vector = Column(BYTEA)
    created_at = Column(DateTime, nullable=False)
    
    audio = relationship("AudioFile", back_populates="features")