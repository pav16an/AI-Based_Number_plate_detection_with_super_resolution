"""Detection services for License Plate Detection System."""

import os
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from src.utils import LicensePlateValidator

logger = logging.getLogger(__name__)


def _import_cv2():
    """Import cv2 lazily so the app can boot without CV dependencies."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for detection features") from exc
    return cv2


def _import_numpy():
    """Import numpy lazily so lightweight code paths remain usable."""
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for detection features") from exc
    return np


class ModelLoader(ABC):
    """Abstract base class for model loading"""
    
    @abstractmethod
    def load(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs):
        """Make predictions"""
        pass


class YOLODetector(ModelLoader):
    """YOLO detector for license plates"""
    
    def __init__(self, model_path: str, device: str = 'cpu', confidence: float = 0.6):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cpu' or 'cuda'
            confidence: Confidence threshold
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.model = None
        self.fallback_model = 'yolov8n.pt'
    
    def load(self) -> bool:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            # Try to load custom model
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Custom YOLO model loaded from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}, using fallback")
                self.model = YOLO(self.fallback_model)
                logger.info(f"Fallback YOLO model loaded: {self.fallback_model}")

            # Fuse layers for faster and slightly more accurate inference if available
            if hasattr(self.model, 'fuse'):
                try:
                    self.model = self.model.fuse()
                    logger.debug("Fused YOLO model layers for optimized inference")
                except Exception:
                    logger.debug("YOLO model fusion not available")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def predict(self, image: Any, conf: Optional[float] = None, iou: float = 0.45, max_det: int = 50) -> List[Dict]:
        """
        Detect license plates in image
        
        Args:
            image: Input image
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold for non-max suppression
            max_det: Maximum number of detections to return
            
        Returns:
            List of detections with bounding boxes and confidence
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            confidence = conf or self.confidence
            results = self.model(image, conf=confidence, iou=iou, max_det=max_det, device=self.device)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().astype(int),
                        'confidence': float(box.conf[0].cpu()),
                        'class_id': int(box.cls[0].cpu())
                    }
                    detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} license plates")
            return detections
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []


class OCREngine(ModelLoader):
    """OCR engine for license plate recognition"""
    
    def __init__(self, language: str = 'en', use_gpu: bool = False):
        """
        Initialize OCR engine
        
        Args:
            language: Language for OCR
            use_gpu: Whether to use GPU
        """
        self.language = language
        self.use_gpu = use_gpu
        self.reader = None
        self.allowed_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.char_substitutions = {
            '0': ['O', 'D', 'Q'],
            '1': ['I', 'L'],
            '2': ['Z'],
            '5': ['S'],
            '6': ['G'],
            '8': ['B'],
            'O': ['0', 'D', 'Q'],
            'D': ['0', 'O'],
            'Q': ['0', 'O'],
            'I': ['1', 'L'],
            'L': ['1', 'I'],
            'Z': ['2'],
            'S': ['5'],
            'G': ['6'],
            'B': ['8'],
        }
    
    def load(self) -> bool:
        """Load OCR model"""
        try:
            import easyocr
            self.reader = easyocr.Reader([self.language], gpu=self.use_gpu)
            logger.info(f"EasyOCR loaded for language: {self.language}")
            return True
        except Exception as e:
            logger.error(f"Error loading OCR model: {e}")
            return False
    
    def predict(self, image: Any, detail: bool = False) -> str:
        """
        Recognize text in image
        
        Args:
            image: Input image (preferably binary/preprocessed)
            detail: Whether to return detailed results
            
        Returns:
            Recognized text or empty string if failed
        """
        if self.reader is None:
            logger.error("OCR model not loaded")
            return ""
        
        try:
            results = self.reader.readtext(image, allowlist=self.allowed_chars, detail=int(detail))
            
            if not results:
                return ""
            
            if detail:
                return results
            
            # Concatenate all recognized text
            text = ''.join([result[1] for result in results])
            return text.upper()
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""
    
    def recognize_with_confidence(self, image: Any) -> Tuple[str, float]:
        """
        Recognize text with confidence score from multiple OCR variants
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            candidates = [image] + ImagePreprocessor.generate_ocr_variants(image)
            candidate_scores = defaultdict(float)
            candidate_confidences: Dict[str, float] = {}
            
            for variant in candidates:
                results = self.predict(variant, detail=True)
                if not results:
                    continue
                
                for bbox, text, conf in results:
                    clean_text = LicensePlateValidator.clean_text(text)
                    if not clean_text:
                        continue

                    for candidate_text in self._generate_candidate_texts(clean_text):
                        score = self._score_candidate(candidate_text, float(conf))
                        if score <= 0:
                            continue
                        candidate_scores[candidate_text] += score
                        candidate_confidences[candidate_text] = max(
                            candidate_confidences.get(candidate_text, 0.0),
                            float(conf),
                        )

            if not candidate_scores:
                return "", 0.0

            best_text = max(
                candidate_scores,
                key=lambda item: (candidate_scores[item], candidate_confidences.get(item, 0.0), len(item)),
            )
            return best_text, candidate_confidences.get(best_text, 0.0)
        except Exception as e:
            logger.error(f"Error in recognition with confidence: {e}")
            return "", 0.0

    def _generate_candidate_texts(self, text: str) -> List[str]:
        """Generate normalized OCR candidates from an ambiguous OCR string."""
        normalized = LicensePlateValidator.clean_text(text)
        if not normalized:
            return []

        candidates = {normalized}
        for index, char in enumerate(normalized):
            for replacement in self.char_substitutions.get(char, []):
                candidate = normalized[:index] + replacement + normalized[index + 1:]
                candidates.add(candidate)

        return list(candidates)

    def _score_candidate(self, text: str, confidence: float) -> float:
        """Score a candidate plate string using OCR confidence and plate heuristics."""
        is_valid, _ = LicensePlateValidator.validate(text)
        score = confidence

        if is_valid:
            score += 1.5
        else:
            score -= 0.75

        length = len(text)
        if 6 <= length <= 10:
            score += 0.35
        elif 4 <= length <= 12:
            score += 0.1
        else:
            score -= 0.4

        letters = sum(1 for char in text if char.isalpha())
        digits = sum(1 for char in text if char.isdigit())

        if letters and digits:
            score += 0.4
        if 1 <= letters <= 4 and 1 <= digits <= 4:
            score += 0.25

        transitions = sum(
            1 for idx in range(1, len(text))
            if text[idx].isalpha() != text[idx - 1].isalpha()
        )
        score += min(transitions * 0.08, 0.32)

        if any(text[idx] == text[idx + 1] == text[idx + 2] for idx in range(max(0, len(text) - 2))):
            score -= 0.4

        return score


class ImagePreprocessor:
    """Preprocessor for license plate images"""
    
    @staticmethod
    def preprocess_for_ocr(image: Any, target_height: int = 64) -> Any:
        """
        Preprocess image for OCR
        
        Args:
            image: Input image
            target_height: Target height for resizing
            
        Returns:
            Preprocessed image
        """
        try:
            cv2 = _import_cv2()
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize preserving aspect ratio
            height, width = gray.shape
            if height < 1 or width < 1:
                return gray
            scale = target_height / height
            new_width = max(1, int(width * scale))
            gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply denoising and contrast enhancement
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Apply adaptive thresholding
            _, binary = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    @staticmethod
    def generate_ocr_variants(image: Any) -> list:
        """
        Generate multiple OCR-friendly variants for the same ROI
        """
        variants = []
        try:
            cv2 = _import_cv2()
            np = _import_numpy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            variants.append(ImagePreprocessor.preprocess_for_ocr(gray, target_height=64))
            variants.append(ImagePreprocessor.enhance_contrast(gray, alpha=1.8, beta=25))
            variants.append(ImagePreprocessor.enhance_contrast(gray, alpha=2.2, beta=10))
            
            # Sharpen image
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(gray, -1, kernel)
            variants.append(ImagePreprocessor.preprocess_for_ocr(sharpened, target_height=64))
            
            # Adaptive thresholding variant
            scaled = cv2.resize(gray, (max(1, gray.shape[1] * 2), max(1, gray.shape[0] * 2)), interpolation=cv2.INTER_CUBIC)
            variants.append(cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 15, 8))

            # Bilateral filtering preserves edges on blurred or low-light plates
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            variants.append(ImagePreprocessor.preprocess_for_ocr(denoised, target_height=72))

            # Inverted threshold can help when the plate is dark-on-light or backlit
            inverted = cv2.bitwise_not(ImagePreprocessor.preprocess_for_ocr(gray, target_height=72))
            variants.append(inverted)
        except Exception as e:
            logger.error(f"Error generating OCR variants: {e}")
        
        return variants
    
    @staticmethod
    def enhance_contrast(image: Any, alpha: float = 1.5, 
                        beta: int = 50) -> Any:
        """
        Enhance image contrast
        
        Args:
            image: Input image
            alpha: Contrast control (1.0 = original)
            beta: Brightness control
            
        Returns:
            Enhanced image
        """
        try:
            cv2 = _import_cv2()
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        except Exception as e:
            logger.error(f"Error enhancing contrast: {e}")
            return image
    
    @staticmethod
    def deskew(image: Any) -> Any:
        """
        Deskew image
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        try:
            cv2 = _import_cv2()
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return image
            
            # Get bounding rectangle of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Rotate image
            h, w = image.shape
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h))
            
            return rotated
        except Exception as e:
            logger.error(f"Error deskewing image: {e}")
            return image


class DetectionService:
    """High-level detection service combining YOLO and OCR"""
    
    def __init__(self, yolo_model_path: str, device: str = 'cpu',
                 ocr_language: str = 'en', use_ocr_gpu: bool = False):
        """Initialize detection service"""
        self.detector = YOLODetector(yolo_model_path, device)
        self.ocr = OCREngine(ocr_language, use_ocr_gpu)
        self.preprocessor = ImagePreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize detector and OCR"""
        detector_ok = self.detector.load()
        ocr_ok = self.ocr.load()
        
        if not detector_ok or not ocr_ok:
            self.logger.error("Failed to initialize detection service")
            return False
        
        self.logger.info("Detection service initialized successfully")
        return True

    @staticmethod
    def _calculate_iou(box1: Any, box2: Any) -> float:
        """Calculate IoU between two xyxy boxes."""
        x1_1, y1_1, x2_1, y2_1 = [int(v) for v in box1]
        x1_2, y1_2, x2_2, y2_2 = [int(v) for v in box2]

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
        box2_area = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
        union = box1_area + box2_area - inter_area

        if union <= 0:
            return 0.0
        return inter_area / union

    def _deduplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
        """Merge overlapping detections from multiple passes."""
        deduped: List[Dict] = []
        for detection in sorted(detections, key=lambda item: item.get('confidence', 0), reverse=True):
            if any(self._calculate_iou(detection['bbox'], existing['bbox']) >= iou_threshold for existing in deduped):
                continue
            deduped.append(detection)
        return deduped

    def _run_detection_passes(self, image: Any, conf: Optional[float] = None) -> List[Dict]:
        """
        Run multiple detector passes to recover small or low-contrast plates.
        """
        base_conf = conf if conf is not None else getattr(self.detector, 'confidence', 0.6)
        detections = self.detector.predict(image, base_conf)
        if detections:
            return detections

        try:
            cv2 = _import_cv2()
        except RuntimeError:
            return detections

        h, w = image.shape[:2]
        retries: List[Tuple[Any, float, int, int, float]] = []

        lower_conf = max(0.25, base_conf - 0.15)
        retries.append((image, lower_conf, 0, 0, 1.0))

        # Upscale the whole image for small plate recovery.
        upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        retries.append((upscaled, lower_conf, 0, 0, 2.0))

        # Focus on the lower-center region where front/rear plates usually appear.
        crop_y1 = int(h * 0.35)
        crop_x1 = int(w * 0.10)
        crop = image[crop_y1:h, crop_x1:int(w * 0.90)]
        if crop.size > 0:
            crop_upscaled = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            retries.append((crop_upscaled, lower_conf, crop_x1, crop_y1, 2.0))

        all_detections: List[Dict] = []
        for retry_image, retry_conf, offset_x, offset_y, scale in retries:
            retry_detections = self.detector.predict(retry_image, retry_conf)
            for detection in retry_detections:
                x1, y1, x2, y2 = [int(v) for v in detection['bbox']]
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                mapped = {
                    **detection,
                    'bbox': [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                }
                all_detections.append(mapped)

        if all_detections:
            self.logger.info("Recovered %s detections from fallback passes", len(all_detections))

        return self._deduplicate_detections(all_detections)
    
    def detect_and_recognize(self, image: Any, 
                           conf: Optional[float] = None) -> List[Dict]:
        """
        Detect and recognize license plates in image
        
        Args:
            image: Input image
            conf: Confidence threshold for detection
            
        Returns:
            List of detections with recognized text
        """
        try:
            detections = self._run_detection_passes(image, conf)
            
            if not detections:
                self.logger.debug("No license plates detected")
                return []
            
            results = []
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                
                # Add a small padding around each detected plate for better OCR
                plate_width = max(1, x2 - x1)
                plate_height = max(1, y2 - y1)
                padding = max(8, int(min(plate_width, plate_height) * 0.08))
                h, w = image.shape[:2]
                x1p = max(0, x1 - padding)
                y1p = max(0, y1 - padding)
                x2p = min(w, x2 + padding)
                y2p = min(h, y2 + padding)
                roi = image[y1p:y2p, x1p:x2p]
                
                if roi is None or roi.size == 0:
                    continue
                
                # Recognize text using multiple OCR variants
                text, ocr_confidence = self.ocr.recognize_with_confidence(roi)

                # Keep the detection even when OCR is uncertain so realtime mode
                # still shows bounding boxes around detected plates.
                result = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'text': text or "",
                    'ocr_confidence': ocr_confidence if text else 0.0
                }
                results.append(result)
            
            self.logger.info(
                "Processed %s detections, OCR resolved %s plates",
                len(results),
                sum(1 for item in results if item.get('text')),
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in detect_and_recognize: {e}")
            return []
