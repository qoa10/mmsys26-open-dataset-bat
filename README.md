# mmsys26-open-dataset-bat

Open bat detection dataset with **annotated images (YOLO format)**, **19 raw videos**, and **baseline scripts** (MOG2 + YOLOv8) for reproducible experiments.

---

## Quick Links

- **Dataset (Google Drive):**  
  https://drive.google.com/drive/folders/1Q2BjR5mpYaQoZ7F73QW6Xd7n1Y_hJ88c?dmr=1&ec=wgc-drive-hero-goto

---

## Dataset Contents

The Drive folder contains two zip files:

- **`Bat Images.zip`** — labeled images for **YOLO object detection**
- **`Bat Videos.zip`** — **19 raw videos** (unlabeled)

---

## Bat Images (YOLO Labeled)

After extracting `Bat Images.zip`, the dataset follows the standard YOLO layout:

```text
Bat Images/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/

### Splits (as reported in the paper)

* **train:** 5010
* **val:** 286
* **test:** 144

> The train split is larger because it includes augmented images.

### `data.yaml`

* Required for YOLO training (paths + class names).
* You may need to **update the dataset root path** inside `data.yaml` based on your local location.
* Please **do not change** class label names/order.

### Augmented vs. Original Images (optional)

* **Original:** filenames start with `frame...`
* **Augmented:** filenames use prefixes (e.g., `blurred_frame...`, `dark_frame...`)

If you want only original data, filter images starting with `frame...` and keep the matching label files.

### Labels

* YOLO `.txt` bounding boxes under `labels/` (object detection, not segmentation).
* If you need COCO JSON or other formats, convert from YOLO using your preferred tool.

---

## Bat Videos (Raw)

After extracting `Bat Videos.zip`, you will find **19 raw videos**:

* Mostly visible-light recordings; filenames use **location + date**
* Suffixes like `_1`, `_2`, `_3` indicate multiple segments from the same session
* Includes **one infrared video** as an additional cross-sensor test case

> Videos are **unlabeled** and intended for benchmarking, testing trained models, or future labeling.

---

## Baseline Code (MOG2 + YOLOv8)

This repository includes baseline scripts for:

* **Video inference** with motion gating (MOG2) + YOLOv8
* **Image-set evaluation** with YOLO labels and IoU matching

### Environment

* **Python:** 3.9+
* **Recommended IDE:** VS Code or PyCharm

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy pandas matplotlib torch
```

> For GPU acceleration, install the CUDA-matching PyTorch build from the official PyTorch site.

### Important: Update Paths Before Running

All scripts currently use **absolute paths** (e.g., `/Users/...`).
Before running, edit each script and update:

* `video_path` / `img_dir` / `label_dir`
* `model_path`
* `save_dir`

---

## Script Overview

### `finaldetect.py` (recommended)

End-to-end **video analysis**:

* MOG2 motion detection + stride-based triggering
* YOLOv8 inference on triggered frames
* Saves structured outputs (CSV) and summary metrics
* Can generate plots (depending on your version)

### `video_mog2+yolov8.py`

Lightweight **video inference** script:

* Motion gating (MOG2) + YOLO stride inference
* Writes an annotated output video + runtime summary

### `yolov8m.py`

YOLO-only **baseline inference** (no MOG2 gating), used for comparison.

### `test_mog2+yolov8m.py`

**Image-set evaluation**:

* Loads YOLO-format GT labels
* Runs YOLOv8 and matches with GT using IoU threshold
* Computes Precision / Recall / F1 and saves annotated images

---

## Run Example

```bash
python finaldetect.py
```

---

## Support

If you have issues accessing the Drive link or downloading the dataset, please open a GitHub issue.

```
::contentReference[oaicite:0]{index=0}
```
