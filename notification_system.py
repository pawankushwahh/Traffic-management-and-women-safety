"""
notification_system.py
=======================
STEP 2 of the pipeline.

This script reads the violations recorded in data/violators_data.csv and,
for each one, sends two emails:

  1. to the vehicle OWNER  (looked up in data/vehicle_database.csv)
  2. to the traffic AUTHORITY for that location (data/location_authorities.csv)

It also saves a text copy of every message in the notification_logs/ folder.

Run it like this:
    python notification_system.py                # uses EMAIL_SIMULATION from .env
    python notification_system.py --simulation   # force "pretend to send" mode
    python notification_system.py --send          # force real email sending
"""

import os
import argparse
import logging
from datetime import datetime

import pandas as pd

import config
from email_service import EmailService


class NotificationSystem:
    def __init__(self, simulation_mode=True):
        self.setup_logging()
        self.simulation_mode = simulation_mode

        # The email helper does the actual sending (or pretending, in simulation).
        self.email_service = EmailService(simulation_mode=simulation_mode)

        if simulation_mode:
            self.logger.info("Running in SIMULATION mode (emails are logged, not sent)")
        else:
            self.logger.info("Running in LIVE mode (emails will really be sent)")

        # Load all three CSV files into memory.
        self.load_datasets()

        # Make sure the folder for saved message copies exists.
        os.makedirs(config.NOTIFICATION_LOGS_DIR, exist_ok=True)

    def setup_logging(self):
        """Print messages to the screen AND save them to a log file."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("notification_system.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    # -----------------------------------------------------------------------
    # Loading the data
    # -----------------------------------------------------------------------

    def load_datasets(self):
        """Read the three CSV files using pandas (dtype=str keeps everything text)."""
        try:
            self.vehicle_db = pd.read_csv(config.VEHICLE_DATABASE_CSV, dtype=str)
            self.violators = pd.read_csv(config.VIOLATORS_CSV, dtype=str)
            self.authorities = pd.read_csv(config.AUTHORITIES_CSV, dtype=str)

            # Tidy up the text so lookups match (remove spaces, fix capitals).
            self.vehicle_db["Vehicle_Number"] = self.vehicle_db["Vehicle_Number"].str.strip().str.upper()
            self.violators["Vehicle_Number"] = self.violators["Vehicle_Number"].str.strip().str.upper()
            self.violators["Location"] = self.violators["Location"].str.strip()
            self.authorities["Location"] = self.authorities["Location"].str.strip()

            self.logger.info(
                "Loaded %s vehicles, %s violations, and %s authorities",
                len(self.vehicle_db), len(self.violators), len(self.authorities),
            )
        except Exception as e:
            self.logger.error("Error loading datasets: %s", e)
            raise

    def get_vehicle_details(self, vehicle_number):
        """Find the owner of a vehicle. Returns a dict, or None if not found."""
        vehicle_number = vehicle_number.strip().upper()
        matches = self.vehicle_db[self.vehicle_db["Vehicle_Number"] == vehicle_number]
        if len(matches) == 0:
            self.logger.warning("Vehicle %s not found in database", vehicle_number)
            return None

        row = matches.iloc[0]
        return {
            "owner_name": row["Owner_Name"],
            "phone": row["Phone_Number"],
            "address": row["Address"],
            "email": row["Owner_Email"],
        }

    def get_authority_details(self, location):
        """Find the traffic authority for a location. Returns a dict, or None."""
        matches = self.authorities[self.authorities["Location"] == location]
        if len(matches) == 0:
            self.logger.warning("No authority found for location: %s", location)
            return None

        row = matches.iloc[0]
        return {
            "name": row["Authority_Name"],
            "phone": row["Authority_Phone"],
            "email": row["Authority_Email"],
        }

    # -----------------------------------------------------------------------
    # Building the email text
    # -----------------------------------------------------------------------

    def compose_violator_message(self, violation, vehicle):
        return (
            f"Traffic Violation Notice\n"
            f"Dear {vehicle['owner_name']},\n"
            f"Your vehicle ({violation['Vehicle_Number']}) "
            f"was detected violating traffic rules:\n"
            f"Violation: {violation['Violation_Type']}\n"
            f"Location: {violation['Location']}\n"
            f"Time: {violation['Violation_Time']}\n"
            f"Fine Amount: Rs. {violation['Fine_Amount']}\n"
            f"Please pay the fine within 7 days to avoid additional penalties."
        )

    def compose_authority_message(self, violation, vehicle):
        return (
            f"New Traffic Violation Detected\n"
            f"Vehicle Number: {violation['Vehicle_Number']}\n"
            f"Violation Type: {violation['Violation_Type']}\n"
            f"Location: {violation['Location']}\n"
            f"Time: {violation['Violation_Time']}\n"
            f"Vehicle Owner: {vehicle['owner_name']}\n"
            f"Owner Contact: {vehicle['phone']}\n"
            f"Owner Email: {vehicle['email']}\n"
            f"Owner Address: {vehicle['address']}\n"
            f"Fine Amount: Rs. {violation['Fine_Amount']}"
        )

    def save_message_copy(self, recipient_type, recipient_details, message):
        """Save a text copy of a message we sent into notification_logs/."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.NOTIFICATION_LOGS_DIR, f"{recipient_type}_{stamp}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Recipient Type: {recipient_type}\n")
            f.write(f"Recipient Details: {recipient_details}\n")
            f.write(f"Message:\n{message}\n")

    # -----------------------------------------------------------------------
    # Sending notifications
    # -----------------------------------------------------------------------

    def process_violation(self, violation):
        """Send the owner and authority emails for one violation row."""
        vehicle_number = violation["Vehicle_Number"]
        self.logger.info("Processing violation for vehicle: %s", vehicle_number)

        # Look up who owns the vehicle.
        vehicle = self.get_vehicle_details(vehicle_number)
        if vehicle is None:
            self.logger.error("Skipping: vehicle %s is not in the database", vehicle_number)
            return False

        # Look up which authority to inform.
        authority = self.get_authority_details(violation["Location"])
        if authority is None:
            self.logger.error("Skipping: no authority for location %s", violation["Location"])
            return False

        # Build the two messages.
        owner_message = self.compose_violator_message(violation, vehicle)
        authority_message = self.compose_authority_message(violation, vehicle)

        owner_subject = f"Traffic Violation Notice - {vehicle_number}"
        authority_subject = f"New Violation Alert - {vehicle_number}"

        # Send them (or simulate sending).
        self.email_service.send_email(vehicle["email"], owner_subject, owner_message)
        self.email_service.send_email(authority["email"], authority_subject, authority_message)

        # Keep a copy on disk.
        self.save_message_copy("violator", vehicle, owner_message)
        self.save_message_copy("authority", authority, authority_message)

        return True

    def process_all_violations(self):
        """Go through every row in the violations CSV."""
        self.logger.info("Starting to process all violations")
        success_count = 0
        total = len(self.violators)

        for _, violation in self.violators.iterrows():
            if self.process_violation(violation):
                success_count += 1

        self.logger.info("Processed %s out of %s violations", success_count, total)
        return success_count, total


def main():
    parser = argparse.ArgumentParser(description="Email traffic violation notices")
    parser.add_argument("--simulation", action="store_true",
                        help="Force simulation mode (no real emails)")
    parser.add_argument("--send", action="store_true",
                        help="Force real email sending")
    args = parser.parse_args()

    # Decide the mode: command-line flags win; otherwise use .env.
    if args.send:
        simulation_mode = False
    elif args.simulation:
        simulation_mode = True
    else:
        simulation_mode = config.EMAIL_SIMULATION

    try:
        notifier = NotificationSystem(simulation_mode=simulation_mode)
        success, total = notifier.process_all_violations()
        print("\nProcessing complete!")
        print(f"Successfully processed {success} out of {total} violations")
        print("Check the notification_logs/ folder for saved copies.")
    except Exception as e:
        print(f"Error running notification system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
