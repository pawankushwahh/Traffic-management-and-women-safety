# Traffic Rule Violation Detection System

This system automatically detects traffic violations (helmet violations and wrong-side driving), extracts number plates, and sends e-challans via SMS.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download YOLOv8 weights:
The system will automatically download YOLOv8n weights on first run.

3. Configure Twilio (for SMS notifications):
- Sign up for a Twilio account at https://www.twilio.com
- Get your Account SID and Auth Token
- Update the `.env` file with your credentials:
  ```
  TWILIO_ACCOUNT_SID=your_account_sid_here
  TWILIO_AUTH_TOKEN=your_auth_token_here
  TWILIO_PHONE_NUMBER=your_twilio_phone_number
  ```

4. Run the system:
```bash
python violation_detector.py
```

## Features
- Real-time detection of helmet violations
- Number plate extraction using EasyOCR
- Automated e-challan generation and SMS notification
- Dummy vehicle database for testing

## Note
This is a basic implementation using pretrained models. For production use, consider:
- Fine-tuning models on your specific use case
- Implementing a proper database
- Adding error handling and logging
- Implementing a web dashboard for monitoring
