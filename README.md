# mmsys26-open-dataset-bat

Open bat detection dataset with annotated images and 19 raw videos, plus metadata and baseline code for reproducible experiments.

---

## Dataset Download (Google Drive)

**Google Drive link:**  
https://drive.google.com/drive/folders/1Q2BjR5mpYaQoZ7F73QW6Xd7n1Y_hJ88c?dmr=1&ec=wgc-drive-hero-goto

The Drive folder contains two zip files:

- `Bat Images.zip` — annotated images for **YOLO object detection**
- `Bat Videos.zip` — **19 raw videos** (unlabeled) for future labeling, benchmarking, or testing

---

## Bat Images (Annotated YOLO Dataset)

After unzipping `Bat Images.zip`, you will see:

- `data.yaml`
- `images/`
- `labels/`

### `data.yaml`

This file is required for training YOLO models. It contains:
- dataset paths
- class labels

**Important:**
- Please **do not change the class label names/order**.
- You may need to **update the dataset path** in `data.yaml` based on your local directory.

### Directory Structure

The dataset follows the standard YOLO layout:
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


### Split Sizes

The split sizes match the paper:

- `train`: **5010**
- `val`: **286**
- `test`: **144**

The training set is larger because it includes **augmented images**.

### Augmented vs. Original Images

- **Original images** use filenames starting with `frame...` (no augmentation prefix).
- **Augmented images** use filenames with prefixes such as:
  - `blurred_frame...`
  - `dark_frame...`
  - (other augmentation-method prefixes)

Naming convention:

- `<augmentation_method> + <original_filename>`

If you want to train only on the original data, select files starting with `frame...` and use the corresponding label files.

### Labels

- Labels are provided as **YOLO `.txt` files** under `labels/`.
- Annotations are **bounding boxes** (object detection), **not segmentation**.

If you require another format (e.g., COCO JSON), please convert the YOLO labels using your preferred conversion tool.

---

## Bat Videos (Raw Videos)

After unzipping `Bat Videos.zip`, you will find **19 videos**:

- **18 standard videos** (visible-light recordings)
  - Naming format: **location + date**
  - Suffixes like `..._1`, `..._2`, `..._3` indicate **multiple segments from the same location/session**.

- **1 infrared video**
  - Included as an additional test case using a different acquisition strategy.
  - Provided for comparison if you are interested in cross-sensor generalization.

### Notes on the Videos

- These videos are **raw data** and have **no annotations**.
- They can be used for:
  - creating new annotations
  - testing trained models
  - future video-based detection/tracking research

---

## Notes / Support

- This repository provides the official dataset structure and usage guidance aligned with our paper.
- If you encounter any access or download issues with the Google Drive link, please open an issue in this repository.



