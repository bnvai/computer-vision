from ultralytics import YOLO
import cv2
import time
import os

# Load pretrained model (nano is fastest)
model = YOLO("../model/yolov8n.pt")

# Load video mp4 (use dataset folder)
video_path = os.path.join("..", "dataset", "traffic.mp4")
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {os.path.abspath(video_path)}")

# COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # car=2, motorcycle=3, bus=5, truck=7

# Per-class BGR colors for better visual distinction
CLASS_COLORS = {
    2: (0, 255, 0),      # car - green
    3: (0, 165, 255),    # motorcycle - orange
    5: (255, 0, 0),      # bus - blue
    7: (255, 0, 255),    # truck - magenta
}

# Target display size for faster processing (keeps aspect ratio)
target_width = 640

# Use video FPS to sync playback; fallback to 30 if missing
src_fps = cap.get(cv2.CAP_PROP_FPS)
delay_ms = max(1, int(1000 / src_fps)) if src_fps > 1 else 33

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]
    new_h = int(h0 * (target_width / w0))
    frame = cv2.resize(frame, (target_width, new_h))

    results = model(frame, imgsz=target_width, conf=0.4, iou=0.5, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes and conf > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls]

            color = CLASS_COLORS.get(cls, (0, 255, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    infer_fps = 1.0 / max(1e-6, time.time() - t0)
    cv2.putText(frame, f"FPS: {infer_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(delay_ms) == 27:
        break

cap.release()
cv2.destroyAllWindows()