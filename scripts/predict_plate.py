"""Run local license plate prediction from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import DetectionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict license plates from a local image using the configured YOLOv10 model."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default="weights/yolov10-license-plate.pt",
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip heavy fallback passes for faster inference",
    )
    parser.add_argument(
        "--save",
        help="Optional output path for the annotated image",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("OpenCV is required to run local predictions.") from exc

    from src.utils import draw_detections

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    service = DetectionService(
        yolo_model_path=args.model,
        device=args.device,
        ocr_language="en",
        use_ocr_gpu=False,
    )
    if not service.initialize():
        raise SystemExit(f"Failed to initialize detection service with model: {args.model}")

    detections = service.detect_and_recognize(
        image,
        conf=args.confidence,
        fast_mode=args.fast,
    )

    payload = [
        {
            "bbox": [int(v) for v in detection["bbox"]],
            "confidence": round(float(detection.get("confidence", 0.0)), 4),
            "text": detection.get("text", ""),
            "ocr_confidence": round(float(detection.get("ocr_confidence", 0.0)), 4),
        }
        for detection in detections
    ]
    print(json.dumps({"image": str(image_path), "detections": payload}, indent=2))

    if args.save:
        annotated = draw_detections(image, detections, ["License"])
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise SystemExit(f"Failed to write annotated image: {output_path}")
        print(f"Annotated image saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
