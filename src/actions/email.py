import os
import smtplib
from email.message import EmailMessage

def send_email(subject: str, body: str) -> str:
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_USER")
    msg["To"] = os.getenv("EMAIL_TO")
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", 587)), timeout=15) as server:
        server.starttls()
        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
        server.send_message(msg)
    return f"sent to {os.getenv('EMAIL_TO')}"

