"""
violation_detector.py
======================
STEP 1 of the pipeline.

This script watches a traffic video and looks for rule violations.
When it is confident a vehicle broke a rule, it:

  1. saves a snapshot image of the vehicle, and
  2. adds one row to data/violators_data.csv

That CSV is the SAME file the notification system reads in step 2, so
detecting a violation here automatically feeds the email step later.

How the detection works (in plain words):
  - YOLO finds vehicles in each video frame.
  - EasyOCR reads the number plate text from the vehicle.
  - Simple rules decide if the vehicle broke a law (e.g. driving on the
    wrong side, or a motorbike with no helmet).

Run it like this:
    python violation_detector.py                       # uses a default video
    python violation_detector.py --video myclip.mp4    # use your own video
    python violation_detector.py --no-display          # don't open a window
"""

import os
import csv
import argparse
from datetime import datetime

import cv2
from ultralytics import YOLO
import easyocr

import config


class ViolationDetector:
    def __init__(self):
        # Load the YOLO model that detects vehicles (cars, bikes, etc.).
        print("Loading YOLO model... (downloads automatically the first time)")
        self.vehicle_model = YOLO(config.YOLO_MODEL)

        # EasyOCR reads text (the number plate) out of an image.
        print("Loading text reader (EasyOCR)...")
        self.reader = easyocr.Reader(["en"])

        # Make sure the folder for snapshot images exists.
        os.makedirs(config.VIOLATIONS_DIR, exist_ok=True)

        # Make sure the violations CSV exists and has a header row.
        self.prepare_violations_csv()

        # --- Memory used while watching the video ---------------------------
        # Plates we have already recorded, so we don't fine the same vehicle twice.
        self.processed_vehicles = set()
        # Counts how many frames each plate has been seen breaking a rule.
        self.tracking_memory = {}
        # Remembers recent positions of each vehicle to work out its direction.
        self.direction_memory = {}

        # The lane regions get set the first time we see a frame (we need the size).
        self.lane_regions = None

        # Drawing settings for the labels we paint on the video.
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    # -----------------------------------------------------------------------
    # CSV handling
    # -----------------------------------------------------------------------

    def prepare_violations_csv(self):
        """Create the violations CSV with a header if it does not exist yet."""
        if not os.path.exists(config.VIOLATORS_CSV):
            with open(config.VIOLATORS_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Vehicle_Number", "Violation_Type", "Location",
                     "Violation_Time", "Fine_Amount"]
                )

    def save_violation(self, plate_text, violation_type, snapshot_image):
        """Save the snapshot image and add one row to the violations CSV."""
        # 1) Save a picture of the offending vehicle.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(
            config.VIOLATIONS_DIR, f"violation_{stamp}_{plate_text}.jpg"
        )
        cv2.imwrite(image_path, snapshot_image)

        # 2) Work out the fine and current time.
        fine = config.get_fine_amount(violation_type)
        violation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3) Add the row that the notification system will read later.
        with open(config.VIOLATORS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [plate_text, violation_type, config.CAMERA_LOCATION,
                 violation_time, fine]
            )

        print(f"Recorded violation: {plate_text} | {violation_type} | Rs.{fine}")
        print(f"   snapshot saved to: {image_path}")

    # -----------------------------------------------------------------------
    # Helper detection logic
    # -----------------------------------------------------------------------

    def setup_lane_regions(self, frame_width):
        """Split the road into a left and right lane down the middle.

        Each lane has an "expected" travel direction. If a vehicle moves the
        other way, we treat it as wrong-side driving. You may need to swap
        'up' and 'down' depending on how your camera is mounted.
        """
        self.lane_regions = {
            "left_lane": {"expected_direction": "up"},
            "right_lane": {"expected_direction": "down"},
        }

    def get_direction(self, plate_text, position):
        """Work out if a vehicle is moving 'up' or 'down' the frame.

        We remember the last few positions of each plate and compare the
        first and last vertical position.
        """
        memory = self.direction_memory.setdefault(plate_text, [])
        memory.append(position)

        # Only keep the last few positions.
        if len(memory) > 3:
            memory.pop(0)

        # Need at least 3 positions to be sure of a direction.
        if len(memory) >= 3:
            first_y = memory[0][1]
            last_y = memory[-1][1]
            if last_y - first_y > 10:
                return "down"
            if first_y - last_y > 10:
                return "up"
        return None

    def is_wrong_side(self, plate_text, position, lane):
        """Return True if the vehicle is travelling against its lane direction."""
        direction = self.get_direction(plate_text, position)
        if direction and lane in self.lane_regions:
            expected = self.lane_regions[lane]["expected_direction"]
            return direction != expected
        return False

    def read_number_plate(self, vehicle_image):
        """Read the number plate text from a cropped vehicle image."""
        try:
            results = self.reader.readtext(vehicle_image)
            for (_box, text, _confidence) in results:
                # Keep only letters/numbers and make it upper case.
                cleaned = "".join(c for c in text if c.isalnum()).upper()
                # A real plate has at least 6 characters and contains a digit.
                if len(cleaned) >= 6 and any(c.isdigit() for c in cleaned):
                    return cleaned
            return None
        except Exception as e:
            print(f"Could not read plate: {e}")
            return None

    # -----------------------------------------------------------------------
    # Processing one video frame
    # -----------------------------------------------------------------------

    def process_frame(self, frame):
        """Look at a single frame, draw boxes, and record any violations."""
        display_frame = frame.copy()
        frame_width = frame.shape[1]

        # Set up the lanes once, the first time we see a frame.
        if self.lane_regions is None:
            self.setup_lane_regions(frame_width)

        # Draw the line that divides the two lanes (just for visualisation).
        cv2.line(display_frame, (frame_width // 2, 0),
                 (frame_width // 2, frame.shape[0]), (255, 255, 0), 2)

        # Ask YOLO to find objects in this frame.
        detections = self.vehicle_model(frame, verbose=False)[0]

        for box in detections.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box
            class_id = int(class_id)

            # YOLO class ids: 2=car, 3=motorcycle, 5=bus, 7=truck.
            is_vehicle = class_id in (2, 3, 5, 7)
            if not is_vehicle or confidence < config.MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # The centre point and which lane the vehicle is in.
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            lane = "left_lane" if center_x < frame_width // 2 else "right_lane"

            # Crop just the vehicle out of the frame.
            vehicle_image = frame[y1:y2, x1:x2]
            if vehicle_image.size == 0:
                continue

            # Read the number plate.
            plate_text = self.read_number_plate(vehicle_image)

            # Decide if a rule was broken.
            violation_type = None
            if plate_text and self.is_wrong_side(plate_text, (center_x, center_y), lane):
                violation_type = "Wrong Side"
            elif class_id == 3 and not self.has_helmet(vehicle_image):
                # class_id 3 == motorcycle
                violation_type = "No Helmet"
            elif config.DEMO_MODE and plate_text:
                # DEMO_MODE: we cannot truly classify the violation without
                # specially trained models, so we label any clearly-seen
                # vehicle based on its type. This lets the whole pipeline run
                # end-to-end on a real video. Turn off DEMO_MODE in config.py
                # once real detection models are added.
                violation_type = config.DEMO_VIOLATION_BY_CLASS.get(class_id, "Wrong Side")

            # Choose a box colour: red = violation, green = fine.
            color = (0, 0, 255) if violation_type else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Build the label text shown above the box.
            label = violation_type or "OK"
            if plate_text:
                label += f" | {plate_text}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        self.font, self.font_scale, color, self.thickness)

            # If we saw a violation with a readable plate, count it.
            if violation_type and plate_text:
                self.track_violation(plate_text, violation_type, vehicle_image)

        return display_frame

    def has_helmet(self, vehicle_image):
        """Very simple helmet check.

        NOTE FOR STUDENTS: doing this properly needs a YOLO model that was
        trained specifically on helmet images. The default yolov8n model was
        NOT trained for helmets, so this is only a placeholder that assumes a
        helmet is present. Replace this with a real helmet model to make
        "No Helmet" detection accurate.
        """
        return True

    def track_violation(self, plate_text, violation_type, vehicle_image):
        """Only record a violation after seeing the same vehicle several times."""
        # Skip if we already fined this vehicle.
        if plate_text in self.processed_vehicles:
            return

        # Count how many frames we have seen this plate breaking a rule.
        record = self.tracking_memory.setdefault(
            plate_text, {"count": 0, "violation_type": violation_type, "image": vehicle_image}
        )
        record["count"] += 1
        record["image"] = vehicle_image

        # Once we are sure enough, save it and stop tracking it.
        if record["count"] >= config.TRACKING_THRESHOLD:
            self.save_violation(plate_text, record["violation_type"], record["image"])
            self.processed_vehicles.add(plate_text)
            del self.tracking_memory[plate_text]

    # -----------------------------------------------------------------------
    # Processing the whole video
    # -----------------------------------------------------------------------

    def process_video(self, video_path, show_window=True):
        """Open the video and process every frame."""
        print(f"\nProcessing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: could not open the video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        if show_window:
            cv2.namedWindow("Traffic Violation Detection", cv2.WINDOW_NORMAL)

        try:
            while True:
                got_frame, frame = cap.read()
                if not got_frame:
                    break  # reached the end of the video

                frame_count += 1
                if total_frames > 0 and frame_count % 30 == 0:
                    percent = (frame_count / total_frames) * 100
                    print(f"Progress: {percent:.0f}%")

                processed = self.process_frame(frame)

                if show_window:
                    cv2.imshow("Traffic Violation Detection", processed)
                    # Press 'q' to stop early.
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

        print("\nDetection complete!")
        print(f"Violations are saved in: {config.VIOLATORS_CSV}")


def find_default_video():
    """Look for a video.mp4 in the project folder, then its parent folder."""
    candidates = [
        os.path.join(config.BASE_DIR, "video.mp4"),
        os.path.join(os.path.dirname(config.BASE_DIR), "video.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]  # default location even if it does not exist yet


def main():
    parser = argparse.ArgumentParser(description="Detect traffic violations in a video")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without opening a video window")
    args = parser.parse_args()

    video_path = args.video or find_default_video()

    detector = ViolationDetector()
    detector.process_video(video_path, show_window=not args.no_display)


if __name__ == "__main__":
    main()
