# Team Tech Vanguard 
## Pawan kushwah
## Rakshita K Biradar 
## Neeraj parmar 
## Ravi Rajpoot 
## Nirmal Mewada

## Women Safety SOS https://github.com/RAVI-RAJPUT-UMATH/Women_Safety : A quick emergency response system for women's safety.
## Signal Automation https://github.com/pawankushwahh/Signal_Automation : An AI-driven system for optimizing traffic signals based on real-time traffic density.
## Traffic Rule Violation Detection and Notification System** (This repository): Automated detection of traffic rule violations using computer vision.
## Video Demonstration https://drive.google.com/file/d/1GaEdtEzO_qE81oPrjO9-C9Vlkk_zl1CV/view?usp=sharing 



# Traffic Rule Violation Detection and Notification System

An automated system that detects traffic rule violations using computer vision, processes the violations, and sends notifications to both violators and traffic authorities.


## Features

- **Real-time Violation Detection**: Uses YOLOv8 for detecting vehicles and identifying violations
- **Multiple Violation Types**: Detects various violations including:
  - No Helmet
  - Wrong Side Driving
  - Signal Jump
- **Automated Notifications**: Sends instant email notifications to:
  - Vehicle owners
  - Traffic authorities
- **Violation Logging**: Maintains detailed logs of all violations
- **Fine Generation**: Automatically calculates fines based on violation type

## How It Works (the flow)

The project runs in two simple steps. They are connected through one CSV file:

```
 video.mp4
    │
    ▼
[ STEP 1: violation_detector.py ]
    - YOLO finds vehicles, EasyOCR reads number plates
    - decides if a rule was broken
    - writes each violation as a row in  data/violators_data.csv
    │
    ▼
 data/violators_data.csv   ← the link between the two steps
    │
    ▼
[ STEP 2: notification_system.py ]
    - reads each violation row
    - looks up the owner (vehicle_database.csv) and police zone (location_authorities.csv)
    - emails both, and saves a copy in notification_logs/
```

You can run both steps at once with `main.py`.

## Prerequisites

1. Python 3.8 or higher
2. A Gmail account (or any SMTP provider) for sending emails
   - You can also run in **simulation mode** with no email account at all.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pawankushwahh/traffic-violation-detection.git
   cd traffic-violation-detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file (copy from `.env.example`) and fill in your values:
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your.email@gmail.com
   SMTP_PASSWORD=your_gmail_app_password
   FROM_EMAIL=your.email@gmail.com
   EMAIL_SIMULATION=true
   ```

   **Gmail setup:** Enable 2-Step Verification, then create an [App Password](https://myaccount.google.com/apppasswords) and use that as `SMTP_PASSWORD`.
   Keep `EMAIL_SIMULATION=true` while testing so no real emails are sent.

## Project Structure

```
Traffic-management-and-women-safety/
├── config.py                     # ALL settings live here (read this first!)
├── main.py                       # run the whole pipeline (step 1 + step 2)
├── violation_detector.py         # STEP 1: detect violations in a video
├── notification_system.py        # STEP 2: email owners and authorities
├── email_service.py              # helper that actually sends the emails
├── data/
│   ├── vehicle_database.csv      # owner info (name, phone, address, email)
│   ├── violators_data.csv        # detected violations (the link between steps)
│   └── location_authorities.csv  # police zone contact details
├── violations/                   # snapshot images of caught vehicles
├── notification_logs/            # a text copy of every email
├── .env.example                  # template for your .env file
└── requirements.txt              # project dependencies
```

## Usage

### Easiest: run everything at once
```bash
python main.py --video video.mp4
```

If you don't want a video window to open (e.g. on a server):
```bash
python main.py --video video.mp4 --no-display
```

### Or run the two steps separately

1. **Detect violations** (writes rows into `data/violators_data.csv`):
   ```bash
   python violation_detector.py --video video.mp4
   ```

2. **Send notifications** for everything recorded so far:
   ```bash
   python notification_system.py
   ```

### Simulation vs. real emails

- By default (`EMAIL_SIMULATION=true`), emails are only printed and saved to
  `notification_logs/` — nothing is actually sent. Great for testing.
- To send for real, set `EMAIL_SIMULATION=false` in `.env`, **or** use the
  `--send` flag:
  ```bash
  python main.py --notify-only --send
  python notification_system.py --send
  ```

### Data File Formats

1. **data/vehicle_database.csv**:
   ```
   Vehicle_Number,Owner_Name,Phone_Number,Address,Owner_Email
   UP32UV1111,Aditya Mishra,+91XXXXXXXXXX,"45 Indira Nagar, Lucknow",aditya.mishra@example.com
   ```
   (Addresses that contain a comma must be wrapped in "double quotes".)

2. **data/violators_data.csv**:
   ```
   Vehicle_Number,Violation_Type,Location,Violation_Time,Fine_Amount
   UP32UV1111,No Helmet,Hazratganj,2025-02-15 08:20:00,500
   ```

## Notes / Known Limitations (for students)

- **Helmet detection is a placeholder.** The default `yolov8n` model was not
  trained on helmets, so `has_helmet()` in `violation_detector.py` currently
  always returns `True`. To make "No Helmet" work, plug in a helmet-trained
  YOLO model there.
- **Detected plates must exist in `vehicle_database.csv`** for an email to be
  sent. If a real-world plate is read that isn't in the database, the
  notification step logs a warning and skips it (this is expected).
- The sample rows already in `data/violators_data.csv` let you test the email
  step on their own, without running detection.

## Notification Format

1. **Violator Notification**:
   ```
   Traffic Violation Notice
   Dear [Owner Name],
   Your vehicle ([Vehicle Number]) was detected violating traffic rules:
   Violation: [Violation Type]
   Location: [Location]
   Time: [Timestamp]
   Fine Amount: Rs. [Amount]
   ```

2. **Authority Notification**:
   ```
   New Traffic Violation Detected
   Vehicle Number: [Number]
   Violation Type: [Type]
   Location: [Location]
   Time: [Timestamp]
   Vehicle Owner: [Name]
   Owner Contact: [Phone]
   Fine Amount: Rs. [Amount]
   ```

## Troubleshooting

1. **No Notifications Being Sent**:
   - Check if SMTP credentials are correctly set in `.env`
   - For Gmail, use an App Password (not your regular password)
   - Verify vehicle exists in database with a valid `Owner_Email`
   - Check network connectivity

2. **Vehicle Not Found**:
   - Ensure vehicle information is present in `vehicle_database.csv`
   - Check if vehicle number format matches database format

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

mail - Kushwahpawan2005@gmail.com
Project Link: https://github.com/pawankushwahh/traffic-violation-detection
