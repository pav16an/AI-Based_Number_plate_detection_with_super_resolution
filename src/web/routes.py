"""Browser-facing routes and compatibility endpoints."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src.db import DetectionRecord
from src.utils import LicensePlateValidator, draw_detections, get_logger

logger = get_logger(__name__)

web = Blueprint("web", __name__)


def _should_expose_error_details() -> bool:
    """Expose diagnostic details outside production only when explicitly enabled."""
    if current_app.config.get("DEBUG"):
        return True
    return os.environ.get("EXPOSE_ERROR_DETAILS", "False").lower() == "true"


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for image processing routes") from exc
    return cv2


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for image processing routes") from exc
    return np


def _ensure_db() -> None:
    """Attach the database manager lazily (lightweight — always safe)."""
    if not hasattr(current_app, "db_manager"):
        from src.db import DatabaseManager
        current_app.db_manager = DatabaseManager(current_app.config["DATABASE_PATH"])


def _ensure_runtime_services() -> None:
    """Attach all runtime services (DB + detection model) lazily."""
    _ensure_db()

    if not hasattr(current_app, "detection_service"):
        from src.services import DetectionService

        svc = DetectionService(
            yolo_model_path=current_app.config["MODEL_PATH"],
            device=current_app.config["MODEL_DEVICE"],
            ocr_language=current_app.config["OCR_LANGUAGE"],
            use_ocr_gpu=current_app.config["OCR_USE_GPU"],
        )
        if not svc.initialize():
            raise RuntimeError(
                "Failed to initialize detection service. "
                "Check that the configured model path exists and all dependencies are installed."
            )
        current_app.detection_service = svc


def _serialize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
    label = detection.get("text") or "License Plate"
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "label": label,
        "confidence": detection.get("confidence", 0),
    }


def _save_valid_detections(detections: List[Dict[str, Any]], source: str, image_path: str | None = None) -> None:
    db_manager = current_app.db_manager
    for detection in detections:
        text = detection.get("text", "")
        is_valid, _ = LicensePlateValidator.validate(text)
        if not is_valid:
            continue

        record = DetectionRecord(
            license_plate=text,
            confidence=detection.get("ocr_confidence", detection.get("confidence", 0)),
            source=source,
            image_path=image_path,
            metadata={"detection_confidence": detection.get("confidence", 0)},
        )
        db_manager.save_detection(record)


@web.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@web.route("/webcam", methods=["GET"])
def webcam():
    return render_template("webcam.html")


@web.route("/results", methods=["GET"])
def results():
    return render_template("results.html")


@web.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        is_allowed = current_app.config["ALLOWED_EXTENSIONS"]
        if not is_allowed:
            return jsonify({"error": "File uploads are disabled"}), 400

        _ensure_runtime_services()
        cv2 = _import_cv2()
        np = _import_numpy()

        filename = secure_filename(file.filename)
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded file is empty"}), 400

        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to decode uploaded image"}), 400

        upload_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)

        detections = current_app.detection_service.detect_and_recognize(
            image,
            conf=max(0.25, current_app.config["DEFAULT_CONFIDENCE"]),
        )
        _save_valid_detections(detections, source="image", image_path=filepath)

        # Best-effort persistence for later review; detection should not fail if this write fails.
        try:
            with open(filepath, "wb") as output_file:
                output_file.write(file_bytes)
        except Exception as save_exc:
            logger.warning("Could not persist uploaded file %s: %s", filepath, save_exc)
            filepath = None

        annotated = draw_detections(image, detections, ["License"])
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            return jsonify({"error": "Failed to encode processed image"}), 500

        return jsonify(
            {
                "image": base64.b64encode(buffer).decode("utf-8"),
                "license_plates": [item.get("text") for item in detections if item.get("text")],
                "detections": [_serialize_detection(item) for item in detections],
                "count": len(detections),
            }
        )
    except RuntimeError as exc:
        logger.error("Upload route unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.error("Upload route failed: %s", exc, exc_info=True)
        payload = {"error": "Image processing failed"}
        if _should_expose_error_details():
            payload["details"] = str(exc)
        return jsonify(payload), 500



@web.route("/upload_video", methods=["POST"])
def upload_video():
    """Process an uploaded video: sample frames, detect plates, return summary."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        video_exts = current_app.config.get("VIDEO_EXTENSIONS", {"mp4", "avi", "mov", "mkv", "webm"})
        if ext not in video_exts:
            return jsonify({"error": f"Unsupported video format: .{ext}"}), 400

        _ensure_runtime_services()
        cv2 = _import_cv2()

        filename = secure_filename(file.filename)
        upload_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
        duration_s = total_frames / fps_video if fps_video > 0 else 0
        max_frames = current_app.config.get("VIDEO_MAX_FRAMES", 60)

        # Evenly sample frames across the video
        if total_frames <= max_frames:
            sample_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            sample_indices = [int(i * step) for i in range(max_frames)]

        all_plates: dict = {}
        all_detections_list = []
        frames_processed = 0
        preview_frame = None
        conf = max(0.25, current_app.config["DEFAULT_CONFIDENCE"])

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            detections = current_app.detection_service.detect_and_recognize(
                frame, conf=conf, fast_mode=True
            )
            frames_processed += 1

            if detections:
                _save_valid_detections(detections, source="video", image_path=filepath)
                timestamp_s = frame_idx / fps_video

                for det in detections:
                    text = det.get("text", "")
                    det_conf = det.get("confidence", 0)
                    ser = _serialize_detection(det)
                    ser["timestamp"] = round(timestamp_s, 2)
                    ser["frame"] = frame_idx
                    all_detections_list.append(ser)
                    if text and (text not in all_plates or det_conf > all_plates[text]):
                        all_plates[text] = det_conf

                if preview_frame is None:
                    annotated = draw_detections(frame, detections, ["License"])
                    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        preview_frame = base64.b64encode(buf).decode("utf-8")

        cap.release()

        return jsonify({
            "success": True,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "duration_seconds": round(duration_s, 2),
            "license_plates": list(all_plates.keys()),
            "plate_confidences": {k: round(v, 3) for k, v in all_plates.items()},
            "detections": all_detections_list,
            "count": len(all_plates),
            "preview": preview_frame,
        })

    except RuntimeError as exc:
        logger.error("Video upload unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.error("Video upload failed: %s", exc, exc_info=True)
        payload = {"error": "Video processing failed"}
        if _should_expose_error_details():
            payload["details"] = str(exc)
        return jsonify(payload), 500

@web.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        if "frame" not in request.files:
            return jsonify({"success": False, "error": "No frame provided"}), 400

        _ensure_runtime_services()
        cv2 = _import_cv2()
        np = _import_numpy()

        uploaded_frame = request.files["frame"]
        frame_bytes = uploaded_frame.read()
        if not frame_bytes:
            return jsonify({"success": False, "error": "Empty frame data"}), 400

        image = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "error": "Failed to decode frame"}), 400

        confidence = request.form.get("confidence", type=float, default=current_app.config["DEFAULT_CONFIDENCE"])
        fast_mode = request.form.get("fast_mode", "false").lower() == "true"
        detections = current_app.detection_service.detect_and_recognize(image, conf=confidence, fast_mode=fast_mode)
        _save_valid_detections(detections, source="webcam")

        return jsonify(
            {
                "success": True,
                "detections": [_serialize_detection(item) for item in detections],
                "license_plates": [item.get("text") for item in detections if item.get("text")],
            }
        )
    except RuntimeError as exc:
        logger.error("Webcam route unavailable: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception as exc:
        logger.error("Webcam route failed: %s", exc, exc_info=True)
        payload = {"success": False, "error": "Frame processing failed"}
        if _should_expose_error_details():
            payload["details"] = str(exc)
        return jsonify(payload), 500


@web.route("/api/plates", methods=["GET"])
def list_plates():
    _ensure_db()
    detections = current_app.db_manager.get_all_detections(limit=500, offset=0)
    return jsonify([item.to_dict() for item in detections]), 200


@web.route("/api/plates/<int:detection_id>", methods=["DELETE"])
def delete_plate(detection_id: int):
    _ensure_db()
    deleted = current_app.db_manager.delete_detection(detection_id)
    if not deleted:
        return jsonify({"success": False, "error": "Record not found"}), 404
    return jsonify({"success": True}), 200


@web.route("/api/plates/bulk-delete", methods=["POST"])
def bulk_delete_plates():
    _ensure_db()
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("ids", [])

    try:
        detection_ids = [int(item) for item in raw_ids]
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid ids supplied"}), 400

    deleted_count = current_app.db_manager.bulk_delete_detections(detection_ids)
    return jsonify({"success": True, "deleted_count": deleted_count}), 200
