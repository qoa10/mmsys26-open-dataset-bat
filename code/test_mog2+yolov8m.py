import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ======================== Configuration ========================
img_dir = "/Users/test/Desktop/test/dataset/images/test"
label_dir = "/Users/test/Desktop/test/dataset/labels/test"
model_path = "/Users/test/Desktop/test/model/best.pt"
save_dir = Path("/Users/test/Desktop/test/results/mog2_yolo")

confidence_threshold = 0.1
iou_threshold = 0.5

# MOG2 parameters (currently not used for gating; kept for future experiments)
min_motion_area = 10
alpha = 0.0  # Temporal smoothing factor for the motion mask (0 means no smoothing)
# ===============================================================

save_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(model_path)

# Background subtractor (kept for potential motion gating/visualization)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=120, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
accumulated_mask = None

# Load images
img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])

# Evaluation counters
total_gt = 0
total_pred = 0
correct = 0

inference_time_total = 0.0
frame_count = 0


def yolo_to_xyxy(label_line: str, img_w: int, img_h: int):
    """
    Convert YOLO format (cls x y w h in normalized coords) to xyxy in pixel coords.
    """
    cls, x, y, w, h = map(float, label_line.split())
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return [x1, y1, x2, y2]


print("Running image evaluation...")

for fname in img_files:
    frame_count += 1
    img_path = os.path.join(img_dir, fname)
    label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        # Skip unreadable files
        continue

    # --- MOG2 motion mask (optional, not used for gating here) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    if accumulated_mask is None:
        accumulated_mask = fgmask.astype(np.float32)
    else:
        accumulated_mask = cv2.addWeighted(
            accumulated_mask, float(alpha), fgmask.astype(np.float32), 1.0 - float(alpha), 0
        )

    _, fgmask_bin = cv2.threshold(accumulated_mask, 50, 255, cv2.THRESH_BINARY)
    fgmask_bin = fgmask_bin.astype(np.uint8)

    contours, _ = cv2.findContours(fgmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(c) > min_motion_area for c in contours)

    # NOTE: This script currently forces detection on every image.
    # If you want true motion gating, replace the line below with: do_infer = motion_detected
    do_infer = True

    # --- Load GT boxes (YOLO txt) ---
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gt_boxes.append(yolo_to_xyxy(line, img.shape[1], img.shape[0]))

    total_gt += len(gt_boxes)

    # --- Run inference and match with GT (IoU) ---
    pred_boxes = []
    results = None

    if do_infer:
        t1 = time.time()
        results = model.predict(img, conf=confidence_threshold, verbose=False)[0]
        inference_time_total += (time.time() - t1)

        # Collect predicted boxes
        if results.boxes is not None and len(results.boxes) > 0:
            pred_boxes = results.boxes.xyxy.cpu().numpy().tolist()
            total_pred += len(pred_boxes)

            # Greedy one-to-one matching: each prediction can match at most one GT
            matched_pred = set()
            for gt in gt_boxes:
                gt_t = torch.tensor([gt], dtype=torch.float32)
                for i, pred in enumerate(pred_boxes):
                    if i in matched_pred:
                        continue
                    pred_t = torch.tensor([pred], dtype=torch.float32)
                    iou = box_iou(gt_t, pred_t)[0, 0].item()
                    if iou >= iou_threshold:
                        correct += 1
                        matched_pred.add(i)
                        break

        # Draw predicted boxes for visualization
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names.get(cls, str(cls))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

    else:
        # Optional: indicate skip (kept for completeness)
        cv2.putText(
            img, "Skipped (no motion)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # Save visualization image
    cv2.imwrite(str(save_dir / fname), img)

# ------------------ Final metrics ------------------
precision = (correct / total_pred) if total_pred else 0.0
recall = (correct / total_gt) if total_gt else 0.0
f1 = (2 * precision * recall / (precision + recall + 1e-8)) if (precision + recall) > 0 else 0.0
avg_infer_time = (inference_time_total / frame_count) if frame_count else 0.0

metrics_file = save_dir / "metrics.txt"
with open(metrics_file, "w", encoding="utf-8") as f:
    f.write(f"Number of images: {frame_count}\n")
    f.write(f"Total GT boxes: {total_gt}\n")
    f.write(f"Total predicted boxes: {total_pred}\n")
    f.write(f"Matched (TP) count: {correct}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1 score:  {f1:.4f}\n")
    f.write(f"Avg inference time per image (s): {avg_infer_time:.4f}\n")

print(f"Done. Results saved to: {save_dir}")
