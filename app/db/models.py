from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import json

Base = declarative_base()

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    chunking_method = Column(String, nullable=False)  # recursive, semantic, custom
    embedding_model = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer, nullable=False)
    
class BookingRequest(Base):
    __tablename__ = "booking_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    booking_date = Column(DateTime, nullable=False)
    booking_time = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, confirmed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)