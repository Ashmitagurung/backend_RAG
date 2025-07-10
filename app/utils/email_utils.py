import smtplib
import ssl

def send_email():
    smtp_server = "smtp.gmail.com"  # example for Gmail SMTP
    port = 465  # SSL port
    sender_email = "your_email@gmail.com"
    password = "your_email_password"
    receiver_email = "receiver@example.com"
    message = """\
Subject: Test Email

This is a test email sent from Python."""

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)

if __name__ == "__main__":
    send_email()
