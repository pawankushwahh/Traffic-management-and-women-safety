"""
main.py
=======
The single entry point that runs the WHOLE pipeline in order:

    Step 1: watch a video and detect violations   (violation_detector.py)
    Step 2: email the owners and authorities       (notification_system.py)

This is the easiest way to run the project.

Examples:
    python main.py                          # detect on the default video, then email
    python main.py --video myclip.mp4       # use your own video
    python main.py --no-display             # don't open a video window
    python main.py --notify-only            # skip detection, just send emails
    python main.py --send                   # actually send emails (not simulation)
"""

import argparse

import config
from violation_detector import ViolationDetector, find_default_video
from notification_system import NotificationSystem


def main():
    parser = argparse.ArgumentParser(description="Run the full traffic violation pipeline")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--no-display", action="store_true",
                        help="Run detection without opening a video window")
    parser.add_argument("--notify-only", action="store_true",
                        help="Skip detection and only send emails for existing violations")
    parser.add_argument("--send", action="store_true",
                        help="Send real emails (default is simulation from .env)")
    args = parser.parse_args()

    # ----- Step 1: detection -------------------------------------------------
    if not args.notify_only:
        print("=" * 60)
        print("STEP 1: Detecting violations in the video")
        print("=" * 60)
        video_path = args.video or find_default_video()
        detector = ViolationDetector()
        detector.process_video(video_path, show_window=not args.no_display)
    else:
        print("Skipping detection (--notify-only).")

    # ----- Step 2: notifications --------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Sending email notifications")
    print("=" * 60)

    # --send overrides the .env setting.
    simulation_mode = config.EMAIL_SIMULATION and not args.send

    notifier = NotificationSystem(simulation_mode=simulation_mode)
    success, total = notifier.process_all_violations()

    print("\nAll done!")
    print(f"Emailed {success} out of {total} recorded violations.")


if __name__ == "__main__":
    main()
