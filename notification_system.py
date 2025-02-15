import pandas as pd
import os
from datetime import datetime
import logging
from twilio.rest import Client
from dotenv import load_dotenv

class NotificationSystem:
    def __init__(self, simulation_mode=False):
        # Initialize logging
        self.setup_logging()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Twilio client
        self.simulation_mode = simulation_mode
        if not simulation_mode:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
            
            self.logger.info(f"Initializing Twilio with Account SID: {account_sid[:6]}...")
            self.logger.info(f"Using Twilio phone number: {self.twilio_phone}")
            
            if not all([account_sid, auth_token, self.twilio_phone]):
                self.logger.error("Missing Twilio credentials in .env file")
                self.logger.error(f"TWILIO_ACCOUNT_SID present: {bool(account_sid)}")
                self.logger.error(f"TWILIO_AUTH_TOKEN present: {bool(auth_token)}")
                self.logger.error(f"TWILIO_PHONE_NUMBER present: {bool(self.twilio_phone)}")
                raise ValueError("Twilio credentials not found")
            
            try:
                self.twilio_client = Client(account_sid, auth_token)
                self.logger.info("Successfully initialized Twilio client")
            except Exception as e:
                self.logger.error(f"Error initializing Twilio client: {e}")
                raise
        
        # Load datasets
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.load_datasets()
        
        # Initialize notification logs directory
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'notification_logs')
        os.makedirs(self.logs_dir, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('notification_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_datasets(self):
        """Load all required datasets"""
        try:
            # Load datasets with proper string handling
            self.vehicle_db = pd.read_csv(os.path.join(self.data_dir, 'vehicle_database.csv'), dtype=str)
            self.violators = pd.read_csv(os.path.join(self.data_dir, 'violators_data.csv'), dtype=str)
            self.authorities = pd.read_csv(os.path.join(self.data_dir, 'location_authorities.csv'), dtype=str)
            
            # Clean whitespace from vehicle numbers and locations
            self.vehicle_db['Vehicle_Number'] = self.vehicle_db['Vehicle_Number'].str.strip()
            self.violators['Vehicle_Number'] = self.violators['Vehicle_Number'].str.strip()
            self.violators['Location'] = self.violators['Location'].str.strip()
            self.authorities['Location'] = self.authorities['Location'].str.strip()
            
            self.logger.info("Successfully loaded all datasets")
            self.logger.info(f"Loaded {len(self.vehicle_db)} vehicles, {len(self.violators)} violations, and {len(self.authorities)} authorities")
        except Exception as e:
            self.logger.error(f"Error loading datasets: {e}")
            raise

    def get_vehicle_details(self, vehicle_number):
        """Retrieve vehicle owner details from the database"""
        try:
            matches = self.vehicle_db[self.vehicle_db['Vehicle_Number'] == vehicle_number]
            if len(matches) == 0:
                self.logger.warning(f"Vehicle {vehicle_number} not found in database")
                return None
            vehicle = matches.iloc[0]
            return {
                'owner_name': vehicle['Owner_Name'],
                'phone': vehicle['Phone_Number'],
                'address': vehicle['Address']
            }
        except Exception as e:
            self.logger.warning(f"Error getting vehicle details for {vehicle_number}: {e}")
            return None

    def get_authority_details(self, location):
        """Get traffic authority details for a location"""
        try:
            matches = self.authorities[self.authorities['Location'] == location]
            if len(matches) == 0:
                self.logger.warning(f"No authority found for location: {location}")
                return None
            authority = matches.iloc[0]
            return {
                'name': authority['Authority_Name'],
                'phone': authority['Authority_Phone'],
                'email': authority['Authority_Email']
            }
        except Exception as e:
            self.logger.warning(f"Error getting authority details for {location}: {e}")
            return None

    def compose_violator_message(self, violation_data, vehicle_details):
        """Compose notification message for violator"""
        return (
            f"Traffic Violation Notice\n"
            f"Dear {vehicle_details['owner_name']},\n"
            f"Your vehicle ({violation_data['Vehicle_Number']}) "
            f"was detected violating traffic rules:\n"
            f"Violation: {violation_data['Violation_Type']}\n"
            f"Location: {violation_data['Location']}\n"
            f"Time: {violation_data['Violation_Time']}\n"
            f"Fine Amount: Rs. {violation_data['Fine_Amount']}\n"
            f"Please pay the fine within 7 days to avoid additional penalties."
        )

    def compose_authority_message(self, violation_data, vehicle_details):
        """Compose notification message for authority"""
        return (
            f"New Traffic Violation Detected\n"
            f"Vehicle Number: {violation_data['Vehicle_Number']}\n"
            f"Violation Type: {violation_data['Violation_Type']}\n"
            f"Location: {violation_data['Location']}\n"
            f"Time: {violation_data['Violation_Time']}\n"
            f"Vehicle Owner: {vehicle_details['owner_name']}\n"
            f"Owner Contact: {vehicle_details['phone']}\n"
            f"Owner Address: {vehicle_details['address']}\n"
            f"Fine Amount: Rs. {violation_data['Fine_Amount']}"
        )

    def send_notification(self, recipient_type, phone_number, message):
        """Send notification via SMS"""
        try:
            if self.simulation_mode:
                self.logger.info(f"SIMULATION: Sending notification to {recipient_type}")
                self.logger.info(f"To: {phone_number}")
                self.logger.info(f"Message:\n{message}\n")
                return True
            else:
                # Send actual SMS using Twilio
                message = self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_phone,
                    to=phone_number
                )
                self.logger.info(f"Sent {recipient_type} notification: {message.sid}")
                return True
        except Exception as e:
            self.logger.error(f"Error sending notification to {phone_number}: {e}")
            return False

    def log_notification(self, recipient_type, recipient_details, message):
        """Log notification details to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"{recipient_type}_{timestamp}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Recipient Type: {recipient_type}\n")
            f.write(f"Recipient Details: {recipient_details}\n")
            f.write(f"Message:\n{message}\n")
        
        self.logger.info(f"Notification logged to {log_file}")

    def process_violation(self, violation_data):
        """Process a single violation and send notifications"""
        try:
            # Get vehicle details
            vehicle_details = self.get_vehicle_details(violation_data['Vehicle_Number'])
            if not vehicle_details:
                self.logger.error(f"Cannot process violation: Vehicle {violation_data['Vehicle_Number']} not found")
                return False

            # Get authority details
            authority_details = self.get_authority_details(violation_data['Location'])
            if not authority_details:
                self.logger.error(f"Cannot process violation: No authority found for {violation_data['Location']}")
                return False

            # Compose messages
            violator_message = self.compose_violator_message(violation_data, vehicle_details)
            authority_message = self.compose_authority_message(violation_data, vehicle_details)

            # Send notifications
            if self.send_notification('violator', vehicle_details['phone'], violator_message):
                self.log_notification('violator', vehicle_details, violator_message)

            if self.send_notification('authority', authority_details['phone'], authority_message):
                self.log_notification('authority', authority_details, authority_message)

            return True
            
        except Exception as e:
            self.logger.error(f"Error processing violation: {e}")
            return False

    def process_all_violations(self):
        """Process all violations in the dataset"""
        self.logger.info("Starting to process all violations")
        success_count = 0
        total_violations = len(self.violators)

        for _, violation in self.violators.iterrows():
            if self.process_violation(violation):
                success_count += 1

        self.logger.info(f"Processed {success_count} out of {total_violations} violations")
        return success_count, total_violations

def main():
    try:
        # Set simulation_mode=True for testing without sending actual SMS
        print("Starting notification system in simulation mode...")
        notification_system = NotificationSystem(simulation_mode=True)
        success, total = notification_system.process_all_violations()
        print(f"\nProcessing complete!")
        print(f"Successfully processed {success} out of {total} violations")
        print(f"Check notification_logs directory for detailed logs")
    except Exception as e:
        print(f"Error running notification system: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
