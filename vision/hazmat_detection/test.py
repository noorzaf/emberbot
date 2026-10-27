#!/usr/bin/env python3
"""
EmberBot — run the trained hazmat model on a single image.

Saves an annotated copy to runs/detect/predict/ and prints each sign found
with its confidence. Only detections at 50% confidence or higher are kept.

Run:  python test.py path/to/image.jpg
"""

import sys
from ultralytics import YOLO

def test_model(image_path):
    """Test the trained model on an image."""
    # Load the trained model
    model = YOLO('model/best.pt')
    
    # Run prediction
    results = model.predict(image_path, save=True, conf=0.5)
    
    print(f"Detection complete! Results saved to: runs/detect/predict/")
    
    # Print detections
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Found {len(boxes)} detections:")
            for box in boxes:
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                print(f"  - {class_name}: {confidence:.2f}")
        else:
            print("No detections found")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_model(image_path)