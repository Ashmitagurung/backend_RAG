# app/utils/email_utils.py

import smtplib
import ssl
from email.message import EmailMessage
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def send_booking_confirmation(receiver_email: str, full_name: str, date: str, time: str) -> bool:
    """
    Sends a booking confirmation email.
    Returns True if successful, False otherwise.
    """
    try:
        smtp_server = settings.SMTP_SERVER
        port = settings.SMTP_PORT
        sender_email = settings.SMTP_SENDER_EMAIL
        password = settings.SMTP_PASSWORD

        subject = "Interview Booking Confirmation"
        body = f"""
Dear {full_name},

Your interview has been successfully booked for:
📅 Date: {date}
⏰ Time: {time}

We look forward to speaking with you!

Best regards,  
Team
        """

        message = EmailMessage()
        message.set_content(body)
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)

        logger.info(f"Confirmation email sent to {receiver_email}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False
