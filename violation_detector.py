import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
from datetime import datetime
import os
from twilio.rest import Client
from dotenv import load_dotenv
import json
import csv

class ViolationDetector:
    def __init__(self):
        # Initialize YOLO model for vehicle and helmet detection
        self.vehicle_model = YOLO('yolov8n.pt')  # Pretrained YOLOv8 model
        self.helmet_model = YOLO('yolov8n.pt')   # We'll use the same model but filter for specific classes
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        
        # Load dummy vehicle database
        self.vehicle_db = self.load_vehicle_database()
        
        # Initialize vehicle tracking
        self.processed_vehicles = set()  # Store processed license plates
        self.tracking_memory = {}  # Store vehicle tracking data
        self.min_detection_confidence = 0.7  # Minimum confidence for detection
        self.tracking_threshold = 10  # Minimum frames to confirm a vehicle
        
        # Initialize Twilio client
        load_dotenv()
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')
        
        # Create violations directory and initialize log file
        self.violations_dir = os.path.join(os.path.dirname(__file__), 'violations')
        os.makedirs(self.violations_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.violations_dir, 'violation_log.csv')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Violation Type', 'Number Plate', 'Image Path'])

    def load_vehicle_database(self):
        """Load dummy vehicle database"""
        try:
            with open('vehicle_database.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create dummy database if not exists
            dummy_db = {
                "MH12DE1234": {
                    "owner_name": "John Doe",
                    "phone": "+1234567890"
                }
            }
            with open('vehicle_database.json', 'w') as f:
                json.dump(dummy_db, f)
            return dummy_db

    def process_frame(self, frame):
        """Process a single frame for violations"""
        # Store original frame
        original_frame = frame.copy()
        
        # Detect vehicles
        vehicle_results = self.vehicle_model(frame, conf=self.min_detection_confidence)[0]
        vehicle_boxes = []
        
        for detection in vehicle_results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) in [2, 3, 5, 7]:  # Filter for cars, motorcycles, buses, and trucks
                vehicle_boxes.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(conf),
                    'cls': int(cls)
                })

        # Process each detected vehicle
        for vehicle in vehicle_boxes:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            # Only process if it's a motorcycle (class 3)
            if vehicle['cls'] != 3:  # Skip if not a motorcycle
                continue
                
            vehicle_img = frame[y1:y2, x1:x2]
            
            if vehicle_img.size == 0:
                continue
            
            # Detect helmet
            helmet_results = self.helmet_model(vehicle_img)[0]
            has_helmet = False
            
            for helmet_detection in helmet_results.boxes.data:
                if int(helmet_detection[5]) == 0:  # Assuming 0 is helmet class
                    has_helmet = True
                    break
            
            # Get license plate
            plate_text = self.detect_license_plate(vehicle_img)
            
            # Draw bounding box
            color = (0, 255, 0) if has_helmet else (0, 0, 255)  # Green for helmet, Red for no helmet
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add text labels
            label = f"Helmet: Yes" if has_helmet else f"No Helmet"
            if plate_text:
                label += f" | Plate: {plate_text}"
            
            # Calculate text position and add background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(original_frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
            cv2.putText(original_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process violation if not already processed
            if plate_text and not has_helmet and plate_text not in self.processed_vehicles:
                if plate_text not in self.tracking_memory:
                    self.tracking_memory[plate_text] = {
                        'count': 1,
                        'best_frame': vehicle_img,
                        'best_conf': vehicle['conf']
                    }
                else:
                    self.tracking_memory[plate_text]['count'] += 1
                    if vehicle['conf'] > self.tracking_memory[plate_text]['best_conf']:
                        self.tracking_memory[plate_text]['best_frame'] = vehicle_img
                        self.tracking_memory[plate_text]['best_conf'] = vehicle['conf']
                
                # Process violation if vehicle has been tracked enough times
                if self.tracking_memory[plate_text]['count'] >= self.tracking_threshold:
                    self.check_and_record_violation(
                        self.tracking_memory[plate_text]['best_frame'],
                        plate_text
                    )
                    self.processed_vehicles.add(plate_text)
                    del self.tracking_memory[plate_text]
        
        return original_frame

    def detect_license_plate(self, vehicle_img):
        """Detect and read license plate from vehicle image"""
        try:
            # Use EasyOCR to detect text
            results = self.reader.readtext(vehicle_img)
            
            # Filter and process results
            for (bbox, text, prob) in results:
                # Basic filtering for license plate format
                text = ''.join(c for c in text if c.isalnum()).upper()
                if len(text) >= 6 and any(c.isdigit() for c in text):
                    return text
            return None
        except Exception as e:
            print(f"Error detecting license plate: {e}")
            return None

    def check_and_record_violation(self, frame, plate_text):
        """Check for violations and record them"""
        # Detect helmet
        helmet_results = self.helmet_model(frame)[0]
        has_helmet = False
        
        for detection in helmet_results.boxes.data:
            cls = int(detection[5])
            if cls == 0:  # Assuming class 0 is for helmet
                has_helmet = True
                break
        
        if not has_helmet:
            # Save violation image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(self.violations_dir, f"violation_{timestamp}_{plate_text}.jpg")
            cv2.imwrite(img_path, frame)
            
            # Log violation
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, 'No Helmet', plate_text, img_path])
            
            # Send notification
            if plate_text in self.vehicle_db:
                self.send_challan(plate_text, img_path)

    def send_challan(self, plate_text, img_path):
        """Send e-challan via SMS"""
        if plate_text in self.vehicle_db:
            owner = self.vehicle_db[plate_text]
            message = f"Traffic Violation Notice:\nVehicle: {plate_text}\nViolation: No Helmet\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nPlease pay the fine within 7 days."
            
            try:
                self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_phone,
                    to=owner['phone']
                )
                return True
            except Exception as e:
                print(f"Error sending SMS: {e}")
                return False
        return False

    def process_video(self, video_path):
        """Process video file for violations"""
        cap = cv2.VideoCapture(video_path)
        
        # Get original video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer
        output_path = os.path.join(os.path.dirname(video_path), 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and get annotated frame
            processed_frame = self.process_frame(frame)
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display frame (for debugging)
            cv2.imshow('Traffic Violation Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Output video saved as: {output_path}")

if __name__ == "__main__":
    detector = ViolationDetector()
    video_path = os.path.join(os.path.dirname(__file__), 'video.mp4')
    print(f"Processing video: {video_path}")
    detector.process_video(video_path)
