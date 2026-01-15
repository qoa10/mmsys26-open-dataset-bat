import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# ======================== Configuration ========================
# Paths (recommended: replace with CLI args later)
video_path = "/Users/test/Desktop/test/videos/Jelena-Capes-8.20.mp4"
model_path = "/Users/test/Desktop/test/model/best.pt"
save_dir = Path("/Users/test/Desktop/test/results/Jelena-Capes-8.20_mog2_yolov8_skip3")

# Inference settings
frame_interval = 3            # Run detection once every N frames (when motion is present)
confidence_threshold = 0.1

# Motion gating (MOG2)
min_motion_area = 10          # Minimum contour area to consider as motion
alpha = 0.0                   # Temporal smoothing factor for motion mask (0 means no smoothing)
# ===============================================================

save_dir.mkdir(parents=True, exist_ok=True)
output_video_path = save_dir / "annotated_video.mp4"
metrics_path = save_dir / "video_metrics.txt"

# ------------------ Setup ------------------
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), f"Cannot open video: {video_path}"

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=120, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
accumulated_mask = None

# Counters
total_frames = 0
yolo_triggered_frames = 0
total_infer_time = 0.0
total_detections = 0

last_results = None

print("Processing video with MOG2 gating + YOLOv8 (stride-based inference)...")

# ------------------ Main loop ------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Motion detection using MOG2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # Morphological dilation to fill small holes
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Optional temporal smoothing for the motion mask
    if accumulated_mask is None:
        accumulated_mask = fgmask.astype("float32")
    else:
        accumulated_mask = cv2.addWeighted(
            accumulated_mask, float(alpha), fgmask.astype("float32"), 1.0 - float(alpha), 0
        )

    _, fgmask_bin = cv2.threshold(accumulated_mask, 50, 255, cv2.THRESH_BINARY)
    fgmask_bin = fgmask_bin.astype("uint8")

    contours, _ = cv2.findContours(fgmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(c) > min_motion_area for c in contours)

    # Trigger YOLO only on selected frames and when motion is detected
    triggered = (total_frames % frame_interval == 1) and motion_detected
    if triggered:
        t1 = time.time()
        last_results = model.predict(frame, conf=confidence_threshold, verbose=False)[0]
        infer_time = time.time() - t1

        total_infer_time += infer_time
        yolo_triggered_frames += 1
        total_detections += len(last_results.boxes)

    # Visualization
    if last_results is not None:
        for box in last_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names.get(cls, str(cls))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    else:
        cv2.putText(
            frame, "Skipped (no motion)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    out.write(frame)

cap.release()
out.release()

avg_infer_time = (total_infer_time / yolo_triggered_frames) if yolo_triggered_frames else 0.0

# ------------------ Write summary ------------------
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Total frames: {total_frames}\n")
    f.write(f"YOLO triggered frames: {yolo_triggered_frames}\n")
    f.write(f"Total detections (sum over triggered frames): {total_detections}\n")
    f.write(f"Total inference time (s): {total_infer_time:.4f}\n")
    f.write(f"Avg inference per triggered frame (s): {avg_infer_time:.4f}\n")

print(f"Completed. Outputs saved to: {save_dir}")
