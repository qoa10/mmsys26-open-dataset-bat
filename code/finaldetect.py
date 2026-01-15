#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script/remote runs
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ======================== Configuration ========================
# Paths (recommended: replace with CLI args later)
video_path = "/Users/test/Desktop/test/batvideo/Cerise - Weston Ranch - Trial 1.mkv"
model_path = "/Users/test/Desktop/test/model/best.pt"
save_dir = Path("/Users/test/Desktop/test/results/Cerise - Weston Ranch - Trial 1.mkv_baseline")

# Inference settings
frame_interval = 10           # Run detection once every N frames (when motion is present)
confidence_threshold = 0.10
imgsz = 640
device = "cpu"                # Use "0" for GPU, "cpu" for CPU
use_half = False              # Half precision (typically GPU only)

# Motion gating (MOG2)
min_motion_area = 10          # Minimum contour area to consider as motion
alpha = 0.0                   # Temporal smoothing factor for motion mask (0 means no smoothing)

# Output options
vid_write = False             # Set True to write an annotated video (may be slow for long videos)
condition_label = "baseline"  # Label used in CSV outputs
time_bin_sec = 60             # Aggregate statistics by time bin (seconds)
# ===============================================================

save_dir.mkdir(parents=True, exist_ok=True)

output_video_path = save_dir / "annotated_video.mp4"
metrics_path = save_dir / "video_metrics.txt"
csv_path = save_dir / "detections.csv"
hist_path = save_dir / "hist.csv"
agg_path = save_dir / "agg_mean_per_frame.csv"
trend_path = save_dir / "trend_minute_mean.csv"

fig_dir = save_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name_no_ext: str):
    """
    Save a figure as both PNG and PDF with tight layout.
    """
    p_png = fig_dir / f"{name_no_ext}.png"
    p_pdf = fig_dir / f"{name_no_ext}.pdf"
    fig.tight_layout()
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p_png, p_pdf


# ------------------ Setup ------------------
model = YOLO(model_path)

cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), f"Cannot open video: {video_path}"

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Optional video writer
if vid_write:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
else:
    out = None

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=120, detectShadows=False
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
accumulated_mask = None

# Counters and logs
total_frames = 0
yolo_triggered_frames = 0
total_infer_time = 0.0
total_detections = 0
records = []

start_wall = time.time()
last_results = None

print(f"Video: {video_path}")
print(f"FPS={fps:.2f}, Resolution={width}x{height}, Frames≈{total_video_frames}")
print("Processing started...")

# ------------------ Main loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    # Timestamp in seconds (fallback to frame index / fps if video timestamp is missing)
    t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    timestamp_s = (t_ms / 1000.0) if (t_ms and t_ms > 0) else (total_frames / fps)

    # Motion detection using MOG2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Optional temporal smoothing for the motion mask
    if accumulated_mask is None:
        accumulated_mask = fgmask.astype("float32")
    else:
        accumulated_mask = cv2.addWeighted(
            accumulated_mask, alpha, fgmask.astype("float32"), 1 - alpha, 0
        )

    _, fgmask_bin = cv2.threshold(accumulated_mask, 50, 255, cv2.THRESH_BINARY)
    fgmask_bin = fgmask_bin.astype("uint8")

    contours, _ = cv2.findContours(
        fgmask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    motion_detected = any(cv2.contourArea(c) > min_motion_area for c in contours)

    # Trigger YOLO only on selected frames and when motion is detected
    triggered = (total_frames % frame_interval == 1) and motion_detected
    if triggered:
        t1 = time.time()
        last_results = model.predict(
            frame,
            imgsz=imgsz,
            conf=confidence_threshold,
            half=use_half,
            device=device,
            verbose=False
        )[0]
        infer_time = time.time() - t1

        total_infer_time += infer_time
        yolo_triggered_frames += 1

        n_boxes = len(last_results.boxes)
        total_detections += n_boxes

        if n_boxes > 0:
            confs = last_results.boxes.conf.detach().cpu().numpy()
            conf_mean = float(np.mean(confs))
            conf_max = float(np.max(confs))
        else:
            conf_mean = 0.0
            conf_max = 0.0

        # Record per-trigger metrics
        records.append({
            "timestamp_s": float(timestamp_s),
            "frame": int(total_frames),
            "bat_count": int(n_boxes),
            "conf_mean": conf_mean,
            "conf_max": conf_max,
            "condition": condition_label
        })

    # Optional annotated video output
    if vid_write and out is not None:
        draw = frame.copy()
        if triggered and (last_results is not None):
            for box in last_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names.get(cls, str(cls))

                cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    draw, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        else:
            cv2.putText(
                draw, "Skipped (no motion or stride)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        out.write(draw)

cap.release()
if out is not None:
    out.release()

# ------------------ Save per-trigger records ------------------
df = pd.DataFrame.from_records(records)
df.to_csv(csv_path, index=False)

# If no frames were triggered, write a minimal metrics file and exit
if len(df) == 0:
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("No triggered frames. Adjust motion threshold or frame_interval.\n")
    raise SystemExit("No triggered frames to analyze.")

# ------------------ Aggregate statistics ------------------
df["minute_bin"] = (df["timestamp_s"] // time_bin_sec).astype(int)

# 1) Bat-minutes: count minutes with at least one detection
bat_min = df.groupby("minute_bin")["bat_count"].max().reset_index(name="max_per_min")
bat_minutes = int((bat_min["max_per_min"] > 0).sum())

# 2) Histogram of detections per triggered frame
hist = df["bat_count"].value_counts().sort_index().reset_index()
hist.columns = ["bat_count", "freq"]
hist.to_csv(hist_path, index=False)

# 3) Mean detections per triggered frame, aggregated by minute
agg = df.groupby("minute_bin")["bat_count"].mean().reset_index(name="mean_bats_per_frame")
agg.to_csv(agg_path, index=False)

# 4) Trend time series (minute index as x-axis)
trend = agg.set_index("minute_bin")
trend.to_csv(trend_path)

# ------------------ Write summary ------------------
avg_infer_time = (total_infer_time / yolo_triggered_frames) if yolo_triggered_frames else 0.0
wall_time = time.time() - start_wall

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("===== Video & Inference Summary =====\n")
    f.write(f"Video path: {video_path}\n")
    f.write(f"FPS: {fps:.2f}, Resolution: {width}x{height}\n")
    f.write(f"Total read frames: {total_frames}\n")
    f.write(f"YOLO triggered frames: {yolo_triggered_frames}\n")
    f.write(f"Total detections (sum over triggered frames): {total_detections}\n")
    f.write(f"Total inference time (s): {total_infer_time:.4f}\n")
    f.write(f"Avg inference per triggered frame (s): {avg_infer_time:.4f}\n")
    f.write(f"Wall clock time (s): {wall_time:.2f}\n")
    f.write("\n===== Derived Metrics =====\n")
    f.write(f"Bat minutes: {bat_minutes}\n")
    f.write(f"Histogram CSV: {hist_path}\n")
    f.write(f"Mean per minute CSV: {agg_path}\n")
    f.write(f"Trend CSV: {trend_path}\n")

print("Inference and aggregation completed. Generating plots...")

# ------------------ Plots (PNG + PDF) ------------------
# A) Histogram: detections per triggered frame
fig = plt.figure(figsize=(6, 4))
x = hist["bat_count"].to_numpy()
y = hist["freq"].to_numpy()
plt.bar(x, y)
plt.xlabel("Bats per frame")
plt.ylabel("Number of frames")
plt.title("Distribution of bats per frame (baseline)")
plt.grid(True, axis="y", alpha=0.3)
p_hist = save_fig(fig, "hist_bats_per_frame_baseline")

# B) Trend: minute-level mean detections
fig = plt.figure(figsize=(7, 4))
plt.plot(trend.index, trend["mean_bats_per_frame"])
plt.xlabel("Time (minute index)")
plt.ylabel("Mean bats per frame")
plt.title("Minute-level mean bats per frame (baseline)")
plt.grid(True, alpha=0.3)
p_trend = save_fig(fig, "trend_mean_bats_per_frame_baseline")

# C) Cumulative bat-minutes over time
bm = (bat_min["max_per_min"] > 0).astype(int).to_numpy()
bm_cum = np.cumsum(bm)
fig = plt.figure(figsize=(7, 4))
plt.step(bat_min["minute_bin"].to_numpy(), bm_cum, where="post")
plt.xlabel("Time (minute index)")
plt.ylabel("Cumulative bat-minutes")
plt.title("Cumulative bat-minutes over time (baseline)")
plt.grid(True, alpha=0.3)
p_bm = save_fig(fig, "bat_minutes_cumulative_baseline")

print("Plots saved:")
print(f" - {p_hist[0]}")
print(f" - {p_trend[0]}")
print(f" - {p_bm[0]}")
print("Done.")
