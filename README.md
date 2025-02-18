# Team Tech Vanguard Presents
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
- **Automated Notifications**: Sends instant notifications to:
  - Vehicle owners (SMS)
  - Traffic authorities (SMS + Email)
- **Violation Logging**: Maintains detailed logs of all violations
- **Fine Generation**: Automatically calculates fines based on violation type

## Prerequisites

1. Python 3.8 or higher
2. Twilio account for SMS notifications
3. CSV files with required data:
   - `vehicle_database.csv`: Vehicle owner information
   - `location_authorities.csv`: Traffic authority contact details
   - `violation_log.csv`: Log of detected violations

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

3. Set up environment variables in `.env` file:
   ```
   TWILIO_ACCOUNT_SID=your_account_sid
   TWILIO_AUTH_TOKEN=your_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone
   ```

## Project Structure

```
rule_1/
├── data/
│   ├── vehicle_database.csv      # Vehicle owner information
│   └── location_authorities.csv  # Authority contact details
├── violations/
│   └── violation_log.csv        # Violation records
├── notification_logs/           # Notification history
├── violation_detector.py        # Main detection script
├── notification_system.py       # Notification handling
└── requirements.txt            # Project dependencies
```

## Usage

1. **Start the Violation Detection System**:
   ```bash
   python violation_detector.py
   ```

2. **Process Violations and Send Notifications**:
   ```bash
   python notification_system.py
   ```

   To test without sending actual notifications:
   ```bash
   python notification_system.py --simulation
   ```

### Data File Formats

1. **vehicle_database.csv**:
   ```
   Vehicle_Number,Owner_Name,Phone_Number,Address
   UP32UV1111,Aditya Mishra,+91XXXXXXXXXX,45 Indira Nagar
   ```

2. **violation_log.csv**:
   ```
   Vehicle_Number,Violation_Type,Location,Violation_Time,Fine_Amount
   UP32UV1111,No Helmet,Hazratganj,2025-02-15 08:20:00,500
   ```

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
   - Check if Twilio credentials are correctly set in `.env`
   - Verify vehicle exists in database
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
