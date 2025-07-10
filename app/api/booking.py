from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime
from typing import List
from app.db.metadata_db import get_db
from app.db.models import BookingRequest
from app.utils.email_utils import send_booking_confirmation

router = APIRouter()

class BookingCreate(BaseModel):
    full_name: str
    email: str
    booking_date: str  # YYYY-MM-DD format
    booking_time: str  # HH:MM format
    notes: str = ""

class BookingResponse(BaseModel):
    id: int
    full_name: str
    email: str
    booking_date: datetime
    booking_time: str
    status: str
    created_at: datetime

@router.post("/book", response_model=BookingResponse)
async def create_booking(booking: BookingCreate, db: Session = Depends(get_db)):
    """Create a new booking"""
    try:
        # Parse date
        booking_date = datetime.strptime(booking.booking_date, "%Y-%m-%d")
        
        # Create booking
        db_booking = BookingRequest(
            full_name=booking.full_name,
            email=booking.email,
            booking_date=booking_date,
            booking_time=booking.booking_time,
            notes=booking.notes
        )
        
        db.add(db_booking)
        db.commit()
        db.refresh(db_booking)
        
        # Send confirmation email
        email_sent = send_booking_confirmation(
            booking.email,
            booking.full_name,
            booking.booking_date,
            booking.booking_time
        )
        
        if not email_sent:
            # Log warning but don't fail the booking
            print(f"Warning: Failed to send confirmation email to {booking.email}")
        
        return db_booking
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bookings", response_model=List[BookingResponse])
async def get_bookings(db: Session = Depends(get_db)):
    """Get all bookings"""
    bookings = db.query(BookingRequest).all()
    return bookings

@router.get("/bookings/{booking_id}", response_model=BookingResponse)
async def get_booking(booking_id: int, db: Session = Depends(get_db)):
    """Get specific booking"""
    booking = db.query(BookingRequest).filter(BookingRequest.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking