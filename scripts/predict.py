"""
Inference script for YOLO defect detection.

Usage:
    # Single image
    python scripts/predict.py --source path/to/image.jpg

    # Directory of images
    python scripts/predict.py --source path/to/images/

    # Using a trained model
    python scripts/predict.py --model runs/train/defect_detection/weights/best.pt --source data/images/test/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO defect detection inference")
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image/directory/video path or URL",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (0-1)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="", help="Device: cpu, 0 (empty=auto)")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated results")
    parser.add_argument("--project", type=str, default="runs/predict", help="Save directory")
    parser.add_argument("--name", type=str, default="exp", help="Run name")
    parser.add_argument("--show", action="store_true", help="Display results in a window")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    print(f"Running inference on: {args.source}")
    print(f"Confidence threshold: {args.conf} | IoU threshold: {args.iou}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device or None,
        save=args.save,
        project=args.project,
        name=args.name,
        show=args.show,
    )

    # Print summary
    total_detections = sum(len(r.boxes) for r in results)
    print(f"\nProcessed {len(results)} image(s) — {total_detections} defect(s) detected")

    for i, r in enumerate(results):
        if len(r.boxes) > 0:
            print(f"  [{i}] {Path(r.path).name}: {len(r.boxes)} defect(s)")
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                print(f"       - {cls_name}: {conf:.2f}")

    if args.save:
        print(f"\nAnnotated results saved to: {results[0].save_dir}")

    return results


if __name__ == "__main__":
    main()
