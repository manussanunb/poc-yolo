"""
Evaluation script — computes mAP, precision, recall on validation/test set.

Usage:
    python scripts/evaluate.py --model runs/train/defect_detection/weights/best.pt
    python scripts/evaluate.py --model runs/train/defect_detection/weights/best.pt --split test
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO defect detection model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/defect_detection.yaml",
        help="Dataset YAML config",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="", help="Device: cpu, 0 (empty=auto)")
    parser.add_argument("--project", type=str, default="runs/val", help="Save directory")
    parser.add_argument("--name", type=str, default="exp", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    print(f"Evaluating model: {args.model}")
    print(f"Dataset: {args.data} | Split: {args.split}")

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )

    print("\n--- Evaluation Results ---")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50-95:   {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")

    return metrics


if __name__ == "__main__":
    main()
