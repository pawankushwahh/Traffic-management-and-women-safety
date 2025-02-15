import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import json
from datetime import datetime
import csv
from dotenv import load_dotenv
from twilio.rest import Client
import pandas as pd

class ViolationDetector:
    def __init__(self):
        # Initialize YOLO models
        self.vehicle_model = YOLO('yolov8n.pt')
        self.helmet_model = YOLO('yolov8n.pt')
        
        # Initialize EasyOCR
        self.reader = easyocr.Reader(['en'])
        
        # Initialize directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.violations_dir = os.path.join(self.base_dir, 'violations')
        os.makedirs(self.violations_dir, exist_ok=True)
        
        # Initialize log files
        self.log_file = os.path.join(self.violations_dir, 'violation_log.csv')
        self.excel_file = os.path.join(self.violations_dir, 'violation_report.xlsx')
        self.initialize_log_files()
        
        # Load vehicle database
        self.vehicle_db = self.load_vehicle_database()
        
        # Initialize tracking parameters
        self.processed_vehicles = set()
        self.tracking_memory = {}
        self.min_detection_confidence = 0.4
        self.tracking_threshold = 5
        
        # Initialize direction detection
        self.direction_memory = {}
        self.direction_threshold = 3
        self.lane_regions = None
        self.wrong_side_vehicles = set()
        
        # Initialize display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Initialize violation data
        self.violations_data = []
        
        # Initialize Twilio
        load_dotenv()
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
        self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER')

    def initialize_log_files(self):
        """Initialize log files with headers"""
        # Initialize CSV log file
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Violation Type', 'Vehicle Number', 'Image Path', 'Location'])
        
        # Initialize Excel file
        if not os.path.exists(self.excel_file):
            df = pd.DataFrame(columns=['Timestamp', 'Violation Type', 'Vehicle Number', 'Image Path', 'Location'])
            df.to_excel(self.excel_file, index=False)

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

    def setup_lane_regions(self, frame_width, frame_height):
        """Setup regions for lane direction detection"""
        # Define regions for different lanes (customize based on your camera view)
        self.lane_regions = {
            'left_lane': {
                'region': (0, 0, frame_width//2, frame_height),
                'expected_direction': 'up'  # or 'down' based on your camera setup
            },
            'right_lane': {
                'region': (frame_width//2, 0, frame_width, frame_height),
                'expected_direction': 'down'  # or 'up' based on your camera setup
            }
        }

    def determine_vehicle_direction(self, vehicle_id, current_position):
        """Determine vehicle movement direction based on position history"""
        if vehicle_id not in self.direction_memory:
            self.direction_memory[vehicle_id] = {
                'positions': [current_position],
                'frames': 1
            }
            return None
        
        # Add new position
        self.direction_memory[vehicle_id]['positions'].append(current_position)
        self.direction_memory[vehicle_id]['frames'] += 1
        
        # Keep only recent positions
        if len(self.direction_memory[vehicle_id]['positions']) > self.direction_threshold:
            self.direction_memory[vehicle_id]['positions'].pop(0)
        
        # Calculate direction if we have enough frames
        if self.direction_memory[vehicle_id]['frames'] >= self.direction_threshold:
            positions = self.direction_memory[vehicle_id]['positions']
            y_coords = [pos[1] for pos in positions]
            
            # Calculate overall movement
            if y_coords[-1] - y_coords[0] > 10:  # Moving down
                return 'down'
            elif y_coords[0] - y_coords[-1] > 10:  # Moving up
                return 'up'
        
        return None

    def check_wrong_side_driving(self, vehicle_id, current_position, lane):
        """Check if vehicle is driving on wrong side"""
        direction = self.determine_vehicle_direction(vehicle_id, current_position)
        
        if direction and lane in self.lane_regions:
            expected_direction = self.lane_regions[lane]['expected_direction']
            return direction != expected_direction
        
        return False

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

    def process_frame(self, frame):
        """Process a single frame for violations"""
        # Store original frame
        original_frame = frame.copy()
        
        # Setup lane regions if not already set
        if self.lane_regions is None:
            self.setup_lane_regions(frame.shape[1], frame.shape[0])
        
        # Draw lane divider (for visualization)
        cv2.line(original_frame, 
                (frame.shape[1]//2, 0), 
                (frame.shape[1]//2, frame.shape[0]), 
                (255, 255, 0), 2)
        
        # Detect vehicles with original dimensions
        vehicle_results = self.vehicle_model(frame)[0]
        
        # Process detections
        for detection in vehicle_results.boxes.data:
            x1, y1, x2, y2, conf, cls = detection
            cls = int(cls)
            
            # Filter for vehicles (car: 2, motorcycle: 3, bus: 5, truck: 7)
            if cls not in [2, 3, 5, 7] or conf < self.min_detection_confidence:
                continue
                
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Calculate vehicle center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Determine which lane the vehicle is in
            lane = 'left_lane' if center_x < frame.shape[1]//2 else 'right_lane'
            
            # Extract vehicle image for processing
            vehicle_img = frame[y1:y2, x1:x2]
            if vehicle_img.size == 0:
                continue
            
            # Initialize violation tracking
            violation_type = None
            has_helmet = True
            
            # Check for helmet violation (only for motorcycles)
            if cls == 3:  # motorcycle
                helmet_results = self.helmet_model(vehicle_img)[0]
                has_helmet = False
                for helmet_det in helmet_results.boxes.data:
                    if int(helmet_det[5]) == 0:  # helmet class
                        has_helmet = True
                        break
            
            # Get license plate
            plate_text = self.detect_license_plate(vehicle_img)
            
            # Check for wrong-side driving
            is_wrong_side = False
            if plate_text:
                is_wrong_side = self.check_wrong_side_driving(plate_text, (center_x, center_y), lane)
            
            # Determine violation type and color
            if is_wrong_side:
                color = (0, 0, 255)  # Red for wrong side
                violation_type = "Wrong Side"
            elif not has_helmet and cls == 3:
                color = (0, 165, 255)  # Orange for no helmet
                violation_type = "No Helmet"
            else:
                color = (0, 255, 0)  # Green for no violation
            
            # Draw bounding box with increased thickness
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label text
            label_parts = []
            if violation_type:
                label_parts.append(violation_type)
            if plate_text:
                label_parts.append(f"Plate: {plate_text}")
            if not label_parts:
                label_parts.append("OK")
            
            label = " | ".join(label_parts)
            
            # Calculate text size and position
            text_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            # Draw text background
            cv2.rectangle(original_frame, 
                        (x1, y1 - 30), 
                        (x1 + text_size[0], y1), 
                        color, -1)
            
            # Draw text
            cv2.putText(original_frame, label, 
                       (x1, y1 - 10), 
                       self.font, self.font_scale, 
                       (255, 255, 255), self.thickness)
            
            # Process violation if detected
            if violation_type and plate_text and plate_text not in self.processed_vehicles:
                if plate_text not in self.tracking_memory:
                    self.tracking_memory[plate_text] = {
                        'count': 1,
                        'best_frame': vehicle_img,
                        'best_conf': conf,
                        'violation_type': violation_type
                    }
                else:
                    self.tracking_memory[plate_text]['count'] += 1
                    if conf > self.tracking_memory[plate_text]['best_conf']:
                        self.tracking_memory[plate_text]['best_frame'] = vehicle_img
                        self.tracking_memory[plate_text]['best_conf'] = conf
                        self.tracking_memory[plate_text]['violation_type'] = violation_type
                
                if self.tracking_memory[plate_text]['count'] >= self.tracking_threshold:
                    self.check_and_record_violation(
                        self.tracking_memory[plate_text]['best_frame'],
                        plate_text,
                        self.tracking_memory[plate_text]['violation_type']
                    )
                    self.processed_vehicles.add(plate_text)
                    del self.tracking_memory[plate_text]
        
        return original_frame

    def check_and_record_violation(self, frame, plate_text, violation_type):
        """Check for violations and record them"""
        # Save violation image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(self.violations_dir, f"violation_{timestamp}_{plate_text}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Record violation data
        violation_data = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Violation Type': violation_type,
            'Vehicle Number': plate_text,
            'Image Path': img_path,
            'Location': 'Main Road'  # You can customize this based on camera location
        }
        
        # Append to violations data
        self.violations_data.append(violation_data)
        
        # Log violation to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                violation_data['Timestamp'],
                violation_data['Violation Type'],
                violation_data['Vehicle Number'],
                violation_data['Image Path'],
                violation_data['Location']
            ])
        
        # Update Excel file
        try:
            df = pd.DataFrame(self.violations_data)
            df.to_excel(self.excel_file, index=False)
        except Exception as e:
            print(f"Error updating Excel file: {e}")
        
        # Send notification
        if plate_text in self.vehicle_db:
            self.send_challan(plate_text, img_path, violation_type)

    def send_challan(self, plate_text, img_path, violation_type):
        """Send e-challan via SMS"""
        if plate_text in self.vehicle_db:
            owner = self.vehicle_db[plate_text]
            message = f"Traffic Violation Notice:\nVehicle: {plate_text}\nViolation: {violation_type}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\nPlease pay the fine within 7 days."
            
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
        print(f"Processing video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Create output video writer
        output_path = os.path.join(os.path.dirname(video_path), 'output_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Create display window
        window_name = 'Traffic Violation Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Calculate window size to maintain aspect ratio
        screen_res = 1920, 1080  # Assuming a common screen resolution
        scale_width = screen_res[0] / frame_width
        scale_height = screen_res[1] / frame_height
        scale = min(scale_width, scale_height)
        
        # Calculate new dimensions
        window_width = int(frame_width * scale)
        window_height = int(frame_height * scale)
        
        # Set window size
        cv2.resizeWindow(window_name, window_width, window_height)
        
        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame to output video
                out.write(processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error processing video: {e}")
        
        finally:
            # Release everything
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Final Excel update
            try:
                if self.violations_data:
                    df = pd.DataFrame(self.violations_data)
                    df.to_excel(self.excel_file, index=False)
                    print(f"\nViolation report saved to: {self.excel_file}")
            except Exception as e:
                print(f"Error saving final Excel report: {e}")
            
            print(f"\nProcessing complete!")
            print(f"Output video saved as: {output_path}")

if __name__ == "__main__":
    detector = ViolationDetector()
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video.mp4')
    detector.process_video(video_path)
