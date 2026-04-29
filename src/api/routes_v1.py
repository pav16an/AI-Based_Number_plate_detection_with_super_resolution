"""
API v1 routes for License Plate Detection
"""

from flask import Blueprint, request, jsonify, current_app, send_from_directory
import logging
import os
from werkzeug.utils import secure_filename
from typing import Tuple

from ..db import DatabaseManager, DetectionRecord
from ..services import DetectionService
from ..utils import (
    LicensePlateValidator,
    ImageValidator,
    get_logger,
    extract_roi,
    draw_detections
)
from .errors import ValidationError, NotFoundError, ServerError

logger = get_logger(__name__)

# Create blueprint
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')


def _serialize_api_detection(detection: dict) -> dict:
    """Convert detection payloads to JSON-safe primitives."""
    bbox = detection.get('bbox', [])
    if hasattr(bbox, 'tolist'):
        bbox = bbox.tolist()

    return {
        'bbox': [int(value) for value in bbox],
        'confidence': float(detection.get('confidence', 0.0)),
        'text': detection.get('text', ''),
        'ocr_confidence': float(detection.get('ocr_confidence', 0.0)),
    }


@api_v1.before_request
def init_services():
    """Initialize shared lightweight services before each request."""
    if not hasattr(current_app, 'db_manager'):
        current_app.db_manager = DatabaseManager(current_app.config['DATABASE_PATH'])


def ensure_detection_service():
    """Initialize the heavy detection service only when a detection route needs it."""
    if hasattr(current_app, 'detection_service'):
        return

    config = current_app.config
    current_app.detection_service = DetectionService(
        yolo_model_path=config['MODEL_PATH'],
        device=config['MODEL_DEVICE'],
        ocr_language=config['OCR_LANGUAGE'],
        use_ocr_gpu=config['OCR_USE_GPU']
    )
    if not current_app.detection_service.initialize():
        raise ServerError("Failed to initialize detection service")


@api_v1.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'version': current_app.config.get('API_VERSION', 'v1'),
            'success': True
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'success': False
        }), 500


@api_v1.route('/detect/image', methods=['POST'])
def detect_image():
    """Detect license plates in uploaded image"""
    try:
        ensure_detection_service()
        # Validate request
        if 'file' not in request.files:
            raise ValidationError("No file provided")
        
        file = request.files['file']
        if file.filename == '':
            raise ValidationError("No file selected")
        
        # Validate file
        if not current_app.config['ALLOWED_EXTENSIONS']:
            raise ValidationError("File uploads not allowed")
        
        is_valid, message = ImageValidator.validate_extension(file.filename)
        if not is_valid:
            raise ValidationError(message)
        
        # Save file
        filename = secure_filename(file.filename)
        upload_dir = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # Load and detect
        import cv2
        image = cv2.imread(filepath)
        if image is None:
            raise ValidationError("Failed to read image file")
        
        detections = current_app.detection_service.detect_and_recognize(image)
        
        # Save results to database
        db_manager = current_app.db_manager
        for detection in detections:
            text = detection.get('text', '')
            conf = detection.get('ocr_confidence', 0)
            
            # Validate license plate
            is_valid, msg = LicensePlateValidator.validate(text)
            if is_valid:
                record = DetectionRecord(
                    license_plate=text,
                    confidence=conf,
                    source='image',
                    image_path=filepath,
                    metadata={'detection_confidence': detection.get('confidence', 0)}
                )
                db_manager.save_detection(record)
        
        return jsonify({
            'detections': [_serialize_api_detection(item) for item in detections],
            'count': len(detections),
            'image_path': filepath,
            'success': True
        }), 200
    
    except ValidationError as e:
        return jsonify({'message': e.message, 'success': False}), 422
    except Exception as e:
        logger.error(f"Error in detect_image: {e}", exc_info=True)
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    """Process frame from webcam"""
    try:
        ensure_detection_service()
        import base64
        import cv2
        import numpy as np
        
        data = request.get_json()
        if not data or 'frame' not in data:
            raise ValidationError("No frame data provided")
        
        # Decode base64 frame
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValidationError("Failed to decode frame")
        
        # Detect
        confidence = float(request.args.get('confidence', current_app.config['DEFAULT_CONFIDENCE']))
        detections = current_app.detection_service.detect_and_recognize(image, conf=confidence)
        
        # Draw and encode result
        annotated = draw_detections(image, detections, ['License'])
        _, buffer = cv2.imencode('.jpg', annotated)
        result_frame = base64.b64encode(buffer).decode()
        
        return jsonify({
            'detections': [_serialize_api_detection(item) for item in detections],
            'frame': f'data:image/jpeg;base64,{result_frame}',
            'count': len(detections),
            'success': True
        }), 200
    
    except ValidationError as e:
        return jsonify({'message': e.message, 'success': False}), 422
    except Exception as e:
        logger.error(f"Error in detect_webcam: {e}", exc_info=True)
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/detections', methods=['GET'])
def get_detections():
    """Get all detections with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        
        if page < 1:
            raise ValidationError("Page must be >= 1")
        if per_page < 1 or per_page > 100:
            raise ValidationError("per_page must be between 1 and 100")
        
        offset = (page - 1) * per_page
        
        db_manager = current_app.db_manager
        detections = db_manager.get_all_detections(limit=per_page, offset=offset)
        
        return jsonify({
            'detections': [d.to_dict() for d in detections],
            'page': page,
            'per_page': per_page,
            'count': len(detections),
            'success': True
        }), 200
    
    except ValidationError as e:
        return jsonify({'message': e.message, 'success': False}), 422
    except Exception as e:
        logger.error(f"Error in get_detections: {e}")
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/detections/<int:detection_id>', methods=['GET'])
def get_detection(detection_id):
    """Get specific detection"""
    try:
        db_manager = current_app.db_manager
        detection = db_manager.get_detection(detection_id)
        
        if not detection:
            raise NotFoundError(f"Detection {detection_id} not found")
        
        return jsonify({
            'detection': detection.to_dict(),
            'success': True
        }), 200
    
    except NotFoundError as e:
        return jsonify({'message': e.message, 'success': False}), 404
    except Exception as e:
        logger.error(f"Error in get_detection: {e}")
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/search', methods=['GET'])
def search_detections():
    """Search detections by license plate"""
    try:
        plate = request.args.get('plate', '')
        if not plate:
            raise ValidationError("plate parameter required")
        
        if len(plate) < 2:
            raise ValidationError("Plate must be at least 2 characters")
        
        db_manager = current_app.db_manager
        detections = db_manager.search_detections(plate)
        
        return jsonify({
            'query': plate,
            'detections': [d.to_dict() for d in detections],
            'count': len(detections),
            'success': True
        }), 200
    
    except ValidationError as e:
        return jsonify({'message': e.message, 'success': False}), 422
    except Exception as e:
        logger.error(f"Error in search_detections: {e}")
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/statistics', methods=['GET'])
def get_statistics():
    """Get statistics"""
    try:
        from datetime import datetime
        
        date = request.args.get('date', datetime.utcnow().date().isoformat())
        
        db_manager = current_app.db_manager
        stats = db_manager.get_statistics(date)
        
        if not stats:
            return jsonify({
                'date': date,
                'statistics': None,
                'success': True
            }), 200
        
        return jsonify({
            'date': date,
            'statistics': stats,
            'success': True
        }), 200
    
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        return jsonify({'message': str(e), 'success': False}), 500


@api_v1.route('/uploads/<filename>', methods=['GET'])
def download_file(filename):
    """Download uploaded file"""
    try:
        upload_dir = current_app.config['UPLOAD_FOLDER']
        filename = secure_filename(filename)
        filepath = os.path.join(upload_dir, filename)
        
        if not os.path.exists(filepath):
            raise NotFoundError("File not found")
        
        return send_from_directory(upload_dir, filename)
    
    except NotFoundError as e:
        return jsonify({'message': e.message, 'success': False}), 404
    except Exception as e:
        logger.error(f"Error in download_file: {e}")
        return jsonify({'message': str(e), 'success': False}), 500
