# Traffic Rule Violation Detection and Notification System

This system automatically detects traffic violations, processes them, and sends notifications to both violators and traffic authorities.

## Features

### 1. Violation Detection
- Detects multiple types of traffic violations:
  - Helmet violations for two-wheelers
  - Wrong-side driving for all vehicles
- Uses YOLOv8 for vehicle detection
- Extracts number plates using EasyOCR

### 2. Notification System
- Processes detected violations automatically
- Matches vehicle numbers with registered vehicle database
- Notifies both violators and traffic authorities
- Generates detailed logs of all notifications

### 3. Data Management
- Maintains three key databases:
  - Vehicle Database (registered vehicles)
  - Violators Database (violation records)
  - Location Authorities Database (traffic authority contacts)

## Setup Instructions

1. Install Dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Twilio (for SMS notifications):
- Sign up for a Twilio account
- Create a `.env` file with your credentials:
```env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone
```

3. Prepare Data Files:
- Ensure the following CSV files are present in the `data` directory:
  - `vehicle_database.csv`
  - `violators_data.csv`
  - `location_authorities.csv`

## Usage

1. Run Violation Detection:
```bash
python violation_detector.py
```

2. Process Violations and Send Notifications:
```bash
python notification_system.py
```

## Output
- Processed video with violations marked
- Notification logs in `notification_logs` directory
- System logs in `notification_system.log`
- Excel reports of violations

## File Structure
```
rule_1/
├── data/
│   ├── vehicle_database.csv
│   ├── violators_data.csv
│   └── location_authorities.csv
├── notification_logs/
├── violations/
├── violation_detector.py
├── notification_system.py
├── requirements.txt
└── README.md
```

## Note
- For production use, consider implementing:
  - Secure database connections
  - API authentication
  - Rate limiting for notifications
  - Backup systems for notification delivery
  - Web interface for monitoring
