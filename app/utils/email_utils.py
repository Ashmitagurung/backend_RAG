import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import settings

def send_booking_confirmation(email: str, full_name: str, date: str, time: str):
    """Send booking confirmation email"""
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = settings.SMTP_USERNAME
    msg['To'] = email
    msg['Subject'] = "Interview Booking Confirmation"
    
    # Email body
    body = f"""
    Dear {full_name},
    
    Your interview has been successfully booked for:
    
    Date: {date}
    Time: {time}
    
    Please make sure to be available at the scheduled time.
    
    If you need to reschedule, please contact us as soon as possible.
    
    Best regards,
    Interview Team
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        server = smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(settings.SMTP_USERNAME, email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False