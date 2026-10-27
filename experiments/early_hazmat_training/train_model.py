# Early version, kept to show how the training setup evolved. See vision/hazmat_detection/ for the final scripts.
from ultralytics import YOLO

# Load a small YOLO model
model = YOLO("yolov8n.pt")

# Train it using your dataset
model.train(data="data.yaml", epochs=30, imgsz=640)  # needs a hand-written data.yaml; no train/val split step yet

print("✅ Training finished! You can now run detection.")
