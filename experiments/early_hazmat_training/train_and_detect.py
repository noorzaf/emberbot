# Early version, kept to show how the training setup evolved. See vision/hazmat_detection/ for the final scripts.
from ultralytics import YOLO
import cv2

# First attempt at training and live detection in one script.
# Train on the converted Hazmat13 data for 10 epochs on the Mac's GPU (mps)
model = YOLO("yolov8n.pt")
model.train(
    data="hazmat13/data.yaml",
    epochs=10,
    imgsz=640,
    device="mps"  # or "cpu" if on Intel Mac
)
print("✅ Training complete!")

# Live detection from the laptop webcam (later replaced by screen-capturing the robot's camera)
model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Hazard Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
