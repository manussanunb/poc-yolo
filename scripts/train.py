"""
Training script for YOLO defect detection.

Usage:
    python scripts/train.py
    python scripts/train.py --model yolo11n.pt --epochs 50 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO for defect detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained model to fine-tune (yolo11n/s/m/l/x.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/defect_detection.yaml",
        help="Path to dataset YAML config",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default="", help="Device: cpu, 0, 0,1 (empty=auto)")
    parser.add_argument("--name", type=str, default="defect_detection", help="Run name")
    parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load pretrained YOLO model (downloads automatically on first run)
    model = YOLO(args.model)

    print(f"Starting training with model: {args.model}")
    print(f"Dataset config: {args.data}")
    print(f"Epochs: {args.epochs} | Image size: {args.imgsz} | Batch: {args.batch}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        # Augmentation (good defaults for defect detection)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("\nTraining complete!")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
