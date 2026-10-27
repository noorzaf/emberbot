#!/usr/bin/env python3
"""
EmberBot — real-time hazmat sign detection on the camera feed.

The robot's Mi Camera 2K streams to its app on the laptop. This script
screen-captures that feed (a 640x600 region at the top-left of the screen),
runs the YOLOv8 model trained on Hazmat13 on every frame, and draws a labelled
box with a confidence score around each hazmat sign it finds.

Run:  python detect_screen.py      (press 'q' in the video window to quit)
Note: transcribed from the team report screenshot; logic unchanged apart from
the model path, which points to the trained weights in this repo.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mss

# Screen region showing the camera app's live feed
monitor = {
    "top": 40,
    "left": 0,
    "width": 640,
    "height": 600,
}


def main():
    # Load the trained YOLO model
    model = YOLO("model/best.pt")

    # Set up screen capture
    sct = mss.mss()

    print("Starting screen capture. Press 'q' to quit.")

    while True:
        # Capture the screen region and convert BGRA -> BGR for OpenCV
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run YOLO; only keep detections with at least 50% confidence
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # Draw a box and "class: confidence" label for each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf.item()
                    class_name = model.names[int(box.cls.item())]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Hazmat Detection (Screen Feed)", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
