"""
config.py
=========
This is the "settings" file for the whole project.

Keeping every setting in ONE place means you only have to look here to
understand (or change) how the system behaves. The other files import
the values from here instead of hard-coding them.

A second-year student reading this project should start by reading this
file top to bottom.
"""

import os
from dotenv import load_dotenv

# Read the .env file (SMTP_USER, SMTP_PASSWORD, ...) into environment variables.
load_dotenv()


# ---------------------------------------------------------------------------
# Where things live on disk
# ---------------------------------------------------------------------------

# The absolute path of the folder that contains this file.
# We build every other path from this so the project works no matter
# which folder you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder that holds our CSV data files.
DATA_DIR = os.path.join(BASE_DIR, "data")

# The three CSV files the project reads.
VEHICLE_DATABASE_CSV = os.path.join(DATA_DIR, "vehicle_database.csv")   # owner info
VIOLATORS_CSV = os.path.join(DATA_DIR, "violators_data.csv")           # detected violations
AUTHORITIES_CSV = os.path.join(DATA_DIR, "location_authorities.csv")   # police contacts

# Folder where snapshot images of caught vehicles are saved.
VIOLATIONS_DIR = os.path.join(BASE_DIR, "violations")

# Folder where a text copy of every email we "send" is saved.
NOTIFICATION_LOGS_DIR = os.path.join(BASE_DIR, "notification_logs")


# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------

# The location label written on every challan.
# IMPORTANT: this name must also appear in data/location_authorities.csv,
# otherwise the system cannot find which police zone to email.
CAMERA_LOCATION = "Hazratganj"

# How much money (in Rupees) to fine for each kind of violation.
FINE_AMOUNTS = {
    "No Helmet": 500,
    "Wrong Side": 1000,
    "Signal Jump": 1500,
}

# Used when a violation type is not listed above.
DEFAULT_FINE = 500

# The YOLO model file used to detect vehicles.
# "yolov8n.pt" is the small, fast version; it downloads automatically
# the first time you run the detector.
YOLO_MODEL = "yolov8n.pt"

# Ignore any detection the model is less than this sure about (0.0 - 1.0).
MIN_CONFIDENCE = 0.4

# A vehicle must be seen breaking the same rule this many times before
# we actually record it. This stops one blurry frame from creating a fine.
TRACKING_THRESHOLD = 5

# --- Demo mode ------------------------------------------------------------
# Detecting the EXACT violation (no helmet, signal jump, ...) accurately
# needs YOLO models that were specially trained for each rule, plus traffic
# signal data. We don't have those here. To still show the full pipeline
# working on a real video, DEMO_MODE flags any clearly-seen vehicle as a
# violation and labels it based on the vehicle type (see below).
# Set this to False once you plug in properly trained detection models.
DEMO_MODE = True

# In DEMO_MODE, which violation label to use for each vehicle type.
# YOLO class ids: 2=car, 3=motorcycle, 5=bus, 7=truck.
DEMO_VIOLATION_BY_CLASS = {
    2: "Wrong Side",   # car
    3: "No Helmet",    # motorcycle
    5: "Wrong Side",   # bus
    7: "Signal Jump",  # truck
}


# ---------------------------------------------------------------------------
# Email settings
# ---------------------------------------------------------------------------

# When True, emails are only printed/logged, NOT actually sent.
# This lets you test the whole project without any email account.
# Set EMAIL_SIMULATION=false in your .env file to send real emails.
EMAIL_SIMULATION = os.getenv("EMAIL_SIMULATION", "true").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------

def get_fine_amount(violation_type):
    """Return the fine for a given violation type (or the default)."""
    return FINE_AMOUNTS.get(violation_type, DEFAULT_FINE)
