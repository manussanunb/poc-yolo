# POC — YOLO Defect Detection

A proof-of-concept for defect detection on images using [Ultralytics YOLO11](https://docs.ultralytics.com/).

## Project Structure

```
poc-yolo/
├── configs/
│   └── defect_detection.yaml   # Dataset config (classes, paths)
├── data/
│   ├── images/
│   │   ├── train/              # Training images
│   │   ├── val/                # Validation images
│   │   └── test/               # Test images
│   └── labels/
│       ├── train/              # YOLO-format labels for train
│       ├── val/                # YOLO-format labels for val
│       └── test/               # YOLO-format labels for test
├── scripts/
│   ├── train.py                # Fine-tune YOLO on your dataset
│   ├── predict.py              # Run inference on images
│   └── evaluate.py             # Compute mAP / precision / recall
├── notebooks/                  # Jupyter notebooks for exploration
├── runs/                       # Training & inference outputs (git-ignored)
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Label Format

YOLO expects one `.txt` file per image in `data/labels/<split>/`.
Each line describes one bounding box:

```
<class_id> <cx> <cy> <width> <height>
```

All values are **normalized** (0–1) relative to image size. Example:

```
0 0.512 0.403 0.120 0.085
2 0.731 0.210 0.055 0.040
```

Use [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/) to annotate and export in YOLO format.

## Define Your Classes

Edit [configs/defect_detection.yaml](configs/defect_detection.yaml) to match your defect types:

```yaml
nc: 3
names:
  0: scratch
  1: dent
  2: crack
```

## Training

```bash
# Fine-tune YOLO11 nano (fastest) on your dataset
python scripts/train.py

# Use a larger model for better accuracy
python scripts/train.py --model yolo11s.pt --epochs 100

# All options
python scripts/train.py --help
```

Model size options (speed vs. accuracy trade-off):

| Model       | Params | Notes                     |
|-------------|--------|---------------------------|
| yolo11n.pt  | 2.6M   | Fastest, least accurate   |
| yolo11s.pt  | 9.4M   | Good balance for POC      |
| yolo11m.pt  | 20M    | Better accuracy           |
| yolo11l.pt  | 25M    | High accuracy             |
| yolo11x.pt  | 56M    | Best accuracy, slowest    |

## Inference

```bash
# Single image
python scripts/predict.py --model runs/train/defect_detection/weights/best.pt \
    --source path/to/image.jpg

# Directory of images
python scripts/predict.py --model runs/train/defect_detection/weights/best.pt \
    --source data/images/test/

# Adjust confidence threshold
python scripts/predict.py --model best.pt --source image.jpg --conf 0.5
```

## Evaluation

```bash
python scripts/evaluate.py --model runs/train/defect_detection/weights/best.pt
```

Outputs: **mAP@50**, **mAP@50-95**, **Precision**, **Recall**.

## Resources

- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/)
- [YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/)
- [Label Studio](https://labelstud.io/) — free annotation tool
- [Roboflow](https://roboflow.com/) — annotation + dataset management
