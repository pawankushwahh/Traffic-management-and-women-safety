import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv


class EmailService:
    def __init__(self, simulation_mode=False):
        load_dotenv()
        self.simulation_mode = simulation_mode
        self.logger = logging.getLogger(__name__)

        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_user)

        if not simulation_mode and not all([self.smtp_user, self.smtp_password, self.from_email]):
            raise ValueError(
                "Email credentials not found. Set SMTP_USER, SMTP_PASSWORD, and FROM_EMAIL in .env"
            )

    def send_email(self, to_email, subject, body):
        """Send a plain-text email via SMTP."""
        if not to_email or not str(to_email).strip():
            self.logger.error("Cannot send email: recipient address is empty")
            return False

        to_email = str(to_email).strip()

        if self.simulation_mode:
            self.logger.info("SIMULATION: Email to %s", to_email)
            self.logger.info("Subject: %s", subject)
            self.logger.info("Message:\n%s\n", body)
            return True

        message = MIMEMultipart()
        message['From'] = self.from_email
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, message.as_string())
            self.logger.info("Email sent to %s", to_email)
            return True
        except Exception as e:
            self.logger.error("Error sending email to %s: %s", to_email, e)
            return False
