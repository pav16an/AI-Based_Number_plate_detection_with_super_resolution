"""
Unit tests for License Plate Detection System
"""

import pytest
import json
import io
import numpy as np
import src.services.detection as detection_module
from src.utils import LicensePlateValidator, ImageValidator
from src.db import DetectionRecord, DatabaseManager
from src.services import DetectionService
from src.services.detection import OCREngine
from src.web.routes import _serialize_detection
import tempfile
import os


class TestValidators:
    """Test validation utilities"""
    
    def test_license_plate_validator_valid(self):
        """Test valid license plate"""
        is_valid, msg = LicensePlateValidator.validate("ABC123")
        assert is_valid, msg
    
    def test_license_plate_validator_too_short(self):
        """Test too short plate"""
        is_valid, msg = LicensePlateValidator.validate("AB")
        assert not is_valid
    
    def test_license_plate_validator_too_long(self):
        """Test too long plate"""
        is_valid, msg = LicensePlateValidator.validate("ABC123456789")
        assert not is_valid
    
    def test_license_plate_validator_no_digits(self):
        """Test plate without digits"""
        is_valid, msg = LicensePlateValidator.validate("ABCDEF")
        assert not is_valid
    
    def test_license_plate_validator_no_letters(self):
        """Test plate without letters"""
        is_valid, msg = LicensePlateValidator.validate("123456")
        assert not is_valid
    
    def test_license_plate_validator_invalid_pattern(self):
        """Test invalid pattern"""
        is_valid, msg = LicensePlateValidator.validate("III000")
        assert not is_valid


class TestDetectionRecord:
    """Test detection record model"""
    
    def test_creation(self):
        """Test creating detection record"""
        record = DetectionRecord(
            license_plate="ABC123",
            confidence=0.95,
            source="image"
        )
        assert record.license_plate == "ABC123"
        assert record.confidence == 0.95
        assert record.source == "image"
    
    def test_to_dict(self):
        """Test converting to dictionary"""
        record = DetectionRecord("ABC123", 0.95)
        data = record.to_dict()
        assert data['license_plate'] == "ABC123"
        assert data['confidence'] == 0.95
    
    def test_to_json(self):
        """Test converting to JSON"""
        record = DetectionRecord("ABC123", 0.95)
        json_str = record.to_json()
        data = json.loads(json_str)
        assert data['license_plate'] == "ABC123"
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            'id': 1,
            'license_plate': 'ABC123',
            'confidence': 0.95,
            'source': 'image',
            'timestamp': '2023-01-01T00:00:00',
            'metadata': {}
        }
        record = DetectionRecord.from_dict(data)
        assert record.license_plate == "ABC123"
        assert record.id == 1


class TestDatabaseManager:
    """Test database manager"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            yield db_path
    
    def test_initialize(self, temp_db):
        """Test database initialization"""
        manager = DatabaseManager(temp_db)
        result = manager.initialize()
        assert result is True
        assert os.path.exists(temp_db)
    
    def test_save_detection(self, temp_db):
        """Test saving detection"""
        manager = DatabaseManager(temp_db)
        manager.initialize()
        
        record = DetectionRecord("ABC123", 0.95)
        result = manager.save_detection(record)
        assert result is True
        assert record.id is not None
    
    def test_get_detection(self, temp_db):
        """Test getting detection"""
        manager = DatabaseManager(temp_db)
        manager.initialize()
        
        # Save record
        record = DetectionRecord("ABC123", 0.95)
        manager.save_detection(record)
        
        # Retrieve record
        retrieved = manager.get_detection(record.id)
        assert retrieved is not None
        assert retrieved.license_plate == "ABC123"
    
    def test_get_all_detections(self, temp_db):
        """Test getting all detections"""
        manager = DatabaseManager(temp_db)
        manager.initialize()
        
        # Save multiple records
        for i in range(5):
            record = DetectionRecord(f"ABC{100+i}", 0.90 + i*0.01)
            manager.save_detection(record)
        
        # Retrieve all
        detections = manager.get_all_detections(limit=10)
        assert len(detections) == 5


@pytest.fixture
def app():
    """Create Flask app for testing"""
    from run import create_app
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


class TestAPI:
    """Test API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
    
    def test_api_health_check(self, client):
        """Test API health check endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_app_info(self, client):
        """Test app info endpoint"""
        response = client.get('/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'version' in data
        assert 'name' in data

    def test_detect_image_serializes_numpy_bboxes(self, app, client, monkeypatch):
        """API image detection should return JSON-safe bbox values."""
        class StubDetectionService:
            def detect_and_recognize(self, image, conf=None, fast_mode=False):
                return [{
                    'bbox': np.array([10, 20, 30, 40]),
                    'confidence': 0.91,
                    'text': 'ABC1234',
                    'ocr_confidence': 0.88,
                }]

        monkeypatch.setattr('src.api.routes_v1.ensure_detection_service', lambda: None)
        app.detection_service = StubDetectionService()

        import cv2
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        assert ok is True

        response = client.post(
            '/api/v1/detect/image',
            data={'file': (io.BytesIO(encoded.tobytes()), 'sample.png')},
            content_type='multipart/form-data'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['detections'][0]['bbox'] == [10, 20, 30, 40]


class TestOCREngineHeuristics:
    """Test OCR post-processing heuristics."""

    def test_generate_candidate_texts_handles_common_confusions(self):
        engine = OCREngine()
        candidates = engine._generate_candidate_texts("ABO123")
        assert "ABO123" in candidates
        assert "AB0123" in candidates

    def test_score_candidate_prefers_valid_plate_pattern(self):
        engine = OCREngine()
        valid_score = engine._score_candidate("AB123CD", 0.70)
        invalid_score = engine._score_candidate("IIIIIII", 0.70)
        assert valid_score > invalid_score


class TestRealtimeDetectionBehavior:
    """Test realtime detection fallback behavior."""

    def test_detection_service_keeps_box_when_ocr_fails(self):
        service = DetectionService("weights/best.pt")

        class StubDetector:
            def predict(self, image, conf=None):
                return [{
                    'bbox': np.array([10, 10, 80, 40]),
                    'confidence': 0.91,
                    'class_id': 0,
                }]

        class StubOCR:
            def recognize_with_confidence(self, roi):
                return "", 0.0

        service.detector = StubDetector()
        service.ocr = StubOCR()

        image = np.zeros((120, 200, 3), dtype=np.uint8)
        detections = service.detect_and_recognize(image)

        assert len(detections) == 1
        assert detections[0]['confidence'] == 0.91
        assert detections[0]['text'] == ""

    def test_serialize_detection_falls_back_to_generic_label(self):
        payload = _serialize_detection({
            'bbox': np.array([1, 2, 3, 4]),
            'confidence': 0.75,
            'text': "",
        })

        assert payload['label'] == "License Plate"

    def test_detection_service_uses_fallback_pass_for_small_plate(self, monkeypatch):
        service = DetectionService("weights/best.pt")

        class FakeCV2:
            INTER_CUBIC = 0

            @staticmethod
            def resize(image, size, interpolation=None):
                target_width, target_height = size
                channels = image.shape[2] if image.ndim == 3 else 1
                if channels == 1:
                    return np.zeros((target_height, target_width), dtype=image.dtype)
                return np.zeros((target_height, target_width, channels), dtype=image.dtype)

        monkeypatch.setattr(detection_module, "_import_cv2", lambda: FakeCV2)

        class StubDetector:
            confidence = 0.6

            def predict(self, image, conf=None):
                height, width = image.shape[:2]
                if width >= 400:
                    return [{
                        'bbox': np.array([40, 20, 120, 60]),
                        'confidence': 0.72,
                        'class_id': 0,
                    }]
                return []

        class StubOCR:
            def recognize_with_confidence(self, roi):
                return "MH20DV2366", 0.88

        service.detector = StubDetector()
        service.ocr = StubOCR()

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = service.detect_and_recognize(image)

        assert len(detections) == 1
        assert detections[0]['text'] == "MH20DV2366"
        assert detections[0]['bbox'] == [20, 10, 60, 30]
