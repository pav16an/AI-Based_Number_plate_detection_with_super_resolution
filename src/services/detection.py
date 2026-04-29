"""Detection services for License Plate Detection System."""

import os
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from src.utils import LicensePlateValidator

logger = logging.getLogger(__name__)

# Fix for Pillow >= 10.0.0 breaking EasyOCR
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS


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


# Ultralytics shims removed (migrated to ONNX Runtime)

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
    """YOLO detector for license plates using pure ONNX Runtime (Zero PyTorch footprint)"""
    
    def __init__(self, model_path: str, device: str = 'cpu', confidence: float = 0.6):
        """
        Initialize YOLO ONNX detector
        
        Args:
            model_path: Path to YOLO model weights (should be .onnx)
            device: 'cpu' or 'cuda'
            confidence: Confidence threshold
        """
        self.model_path = model_path
        if self.model_path.endswith('.pt'):
            self.model_path = self.model_path.replace('.pt', '.onnx')
        self.device = device
        self.confidence = confidence
        self.session = None
    
    def load(self) -> bool:
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            if os.path.exists(self.model_path):
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                logger.info(f"ONNX YOLO model loaded from {self.model_path}")
            else:
                logger.error(f"ONNX model not found at {self.model_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading ONNX YOLO model: {e}")
            return False
    
    def predict(self, image: Any, conf: Optional[float] = None, iou: float = 0.45,
                max_det: int = 50, fast_mode: bool = False) -> List[Dict]:
        """
        Detect license plates in image using ONNX Runtime
        """
        if self.session is None:
            logger.error("ONNX Model not loaded")
            return []
        
        try:
            confidence = conf or self.confidence
            
            # 1. Letterbox resizing to 640x640 (required for this specific exported model)
            import cv2
            import numpy as np
            
            shape = image.shape[:2]
            new_shape = (640, 640)
            r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
            new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
            dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
            dw /= 2
            dh /= 2
            
            if shape[::-1] != new_unpad:
                im = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            else:
                im = image.copy()
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            # 2. Convert HWC to CHW, BGR to RGB
            im = im.transpose((2, 0, 1))[::-1]  
            im = np.ascontiguousarray(im)
            im = im.astype(np.float32) / 255.0
            im = np.expand_dims(im, axis=0)

            # 3. ONNX Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: im})
            preds = outputs[0][0] # YOLOv10 output is (1, 300, 6)
            
            # 4. Parse outputs and scale bounding boxes back
            detections = []
            for pred in preds:
                box_conf = float(pred[4])
                if box_conf < confidence:
                    continue
                
                # Scale boxes back
                x1, y1, x2, y2 = pred[:4]
                x1 = (x1 - dw) / r
                x2 = (x2 - dw) / r
                y1 = (y1 - dh) / r
                y2 = (y2 - dh) / r
                
                # Clip to image boundaries
                x1 = max(0, min(shape[1], x1))
                x2 = max(0, min(shape[1], x2))
                y1 = max(0, min(shape[0], y1))
                y2 = max(0, min(shape[0], y2))

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': box_conf,
                    'class_id': int(pred[5])
                })
            
            # Sort by confidence descending and limit to max_det
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            detections = detections[:max_det]
            
            logger.debug(f"Detected {len(detections)} license plates (ONNX)")
            return detections
        except Exception as e:
            logger.error(f"Error during ONNX detection: {e}")
            return []


class OCREngine(ModelLoader):
    """OCR engine for license plate recognition"""

    # Bounded LRU-style cache: maps crop-hash → (text, confidence)
    _OCR_CACHE_MAX = 128
    
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
        # OCR result cache: crop_hash → (text, confidence)
        self._ocr_cache: Dict[int, Tuple[str, float]] = {}
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
            import pytesseract
            self.reader = pytesseract
            logger.info("Tesseract OCR initialized")
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
            import pytesseract
            from pytesseract import Output
            
            # Configure Tesseract to look for sparse text and restrict characters. 
            # PSM 11 (Sparse text) is far more accurate for multi-line or noisy plates than PSM 7 (single line).
            custom_config = f'-c tessedit_char_whitelist={self.allowed_chars} --psm 11'
            
            if detail:
                data = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)
                results = []
                for i in range(len(data['text'])):
                    conf = float(data['conf'][i])
                    text = data['text'][i].strip()
                    if conf > 0 and text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
                        results.append((bbox, text, conf / 100.0))
                return results
            else:
                text = pytesseract.image_to_string(image, config=custom_config).strip()
                return text.upper()
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return ""

    def _crop_hash(self, image: Any) -> int:
        """Fast hash of a crop array for cache keying."""
        try:
            np = _import_numpy()
            # Downsample to a tiny thumbnail before hashing for speed
            small = image[::4, ::4] if image.ndim == 2 else image[::4, ::4, 0]
            return hash(small.tobytes())
        except Exception:
            return 0
    
    def recognize_with_confidence(self, image: Any, fast_mode: bool = False) -> Tuple[str, float]:
        """
        Recognize text with confidence score from multiple OCR variants.

        fast_mode=True  → 2 variants (original + 1 enhanced).  Fastest, good for video/webcam.
        fast_mode=False → 7 variants.  Best accuracy, used for single-image uploads.
        
        Args:
            image: Input image
            fast_mode: If True, use fewer OCR variants for speed
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            # ── Cache lookup ────────────────────────────────────────────────────
            crop_hash = self._crop_hash(image)
            if crop_hash and crop_hash in self._ocr_cache:
                logger.debug("OCR cache hit")
                return self._ocr_cache[crop_hash]

            # ── Build variant list ──────────────────────────────────────────────
            if fast_mode:
                # 2 variants: original + one contrast-enhanced pass
                try:
                    enhanced = ImagePreprocessor.enhance_contrast(image, alpha=1.5, beta=20)
                    candidates = [image, enhanced]
                except Exception:
                    candidates = [image]
            else:
                candidates = [image] + ImagePreprocessor.generate_ocr_variants(image)

            candidate_scores = defaultdict(float)
            candidate_confidences: Dict[str, float] = {}
            
            for variant in candidates:
                results = self.predict(variant, detail=True)
                if not results:
                    continue
                
                # Combine all text fragments found in this crop variant
                combined_text = "".join([t for _, t, _ in results])
                if not combined_text:
                    continue
                    
                avg_conf = sum([c for _, _, c in results]) / len(results)
                
                clean_text = LicensePlateValidator.clean_text(combined_text)
                if not clean_text:
                    continue

                for candidate_text in self._generate_candidate_texts(clean_text):
                    score = self._score_candidate(candidate_text, avg_conf)
                    if score <= 0:
                        continue
                    candidate_scores[candidate_text] += score
                    candidate_confidences[candidate_text] = max(
                        candidate_confidences.get(candidate_text, 0.0),
                        avg_conf,
                    )

            if not candidate_scores:
                result = ("", 0.0)
            else:
                best_text = max(
                    candidate_scores,
                    key=lambda item: (candidate_scores[item], candidate_confidences.get(item, 0.0), len(item), item),
                )
                result = (best_text, candidate_confidences.get(best_text, 0.0))

            # ── Cache store (bounded) ───────────────────────────────────────────
            if crop_hash:
                if len(self._ocr_cache) >= self._OCR_CACHE_MAX:
                    # Evict oldest entry
                    self._ocr_cache.pop(next(iter(self._ocr_cache)))
                self._ocr_cache[crop_hash] = result

            return result
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

        return sorted(list(candidates))

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

            # Morphological Closing to connect fragmented text (rain/motion blur)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
            variants.append(ImagePreprocessor.preprocess_for_ocr(closing, target_height=64))
        except Exception as e:
            logger.error(f"Error generating OCR variants: {e}")
        
        return variants
    
    @staticmethod
    def gamma_correction(image: Any, gamma: float = 1.0) -> Any:
        """Apply gamma correction for dark/night images"""
        try:
            cv2 = _import_cv2()
            import numpy as np
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        except Exception as e:
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error applying gamma correction: {e}")
            return image

    @staticmethod
    def apply_clahe(image: Any) -> Any:
        """Apply CLAHE to handle fog/rain washout conditions"""
        try:
            cv2 = _import_cv2()
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_channel)
                limg = cv2.merge((cl, a, b))
                return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except Exception as e:
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error applying CLAHE: {e}")
            return image

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

    def _run_detection_passes(self, image: Any, conf: Optional[float] = None, fast_mode: bool = False) -> List[Dict]:
        """
        Run detector passes to find license plates.
        fast_mode: single pass at imgsz=640 — used for video/webcam.
        normal:    multi-pass with fallbacks — used for image uploads.
        """
        base_conf = conf if conf is not None else getattr(self.detector, 'confidence', 0.35)

        all_detections: List[Dict] = []

        # Initial pass — fast_mode controls YOLO imgsz (640 vs 960)
        initial_detections = self.detector.predict(image, base_conf, fast_mode=fast_mode)
        if initial_detections:
            all_detections.extend(initial_detections)

        # In fast_mode, skip expensive fallback passes
        if initial_detections or fast_mode:
            return self._deduplicate_detections(all_detections)

        try:
            cv2 = _import_cv2()
        except RuntimeError:
            return self._deduplicate_detections(all_detections)

        h, w = image.shape[:2]
        retries: List[Tuple[Any, float, int, int, float]] = []

        lower_conf = max(0.25, base_conf - 0.15)
        
        # Full image lower conf
        retries.append((image, lower_conf, 0, 0, 1.0))

        # Upscale the whole image for small plate recovery, ONLY if image is small to prevent OOM
        if w < 1280 and h < 1280:
            upscaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            retries.append((upscaled, lower_conf, 0, 0, 2.0))

        # Contrast-enhanced retry can recover low-contrast or dim plates.
        enhanced = ImagePreprocessor.enhance_contrast(image, alpha=1.35, beta=18)
        retries.append((enhanced, lower_conf, 0, 0, 1.0))
        
        # Gamma correction pass for extremely dark/night images
        gamma_corrected = ImagePreprocessor.gamma_correction(image, gamma=1.5)
        retries.append((gamma_corrected, lower_conf, 0, 0, 1.0))
        
        # CLAHE pass for foggy/rainy/washed-out conditions
        clahe_enhanced = ImagePreprocessor.apply_clahe(image)
        retries.append((clahe_enhanced, lower_conf, 0, 0, 1.0))

        if not initial_detections:
            # Focus on the lower-center region when the first pass misses a plate.
            crop_y1 = int(h * 0.35)
            crop_x1 = int(w * 0.10)
            crop = image[crop_y1:h, crop_x1:int(w * 0.90)]
            if crop.size > 0:
                retries.append((crop, lower_conf, crop_x1, crop_y1, 1.0))
                crop_upscaled = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                retries.append((crop_upscaled, lower_conf, crop_x1, crop_y1, 2.0))

        for retry_image, retry_conf, offset_x, offset_y, scale in retries:
            retry_detections = self.detector.predict(retry_image, retry_conf, fast_mode=False)
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
            self.logger.info(f"Accumulated {len(all_detections)} raw detections across passes")

        return self._deduplicate_detections(all_detections)

    def _recognize_plate_text(self, roi: Any, fast_mode: bool = False) -> Tuple[str, float]:
        """Call OCR with backward compatibility for older test doubles."""
        try:
            return self.ocr.recognize_with_confidence(roi, fast_mode=fast_mode)
        except TypeError as exc:
            if "fast_mode" not in str(exc):
                raise
            return self.ocr.recognize_with_confidence(roi)
    
    def detect_and_recognize(self, image: Any, 
                           conf: Optional[float] = None,
                           fast_mode: bool = False) -> List[Dict]:
        """
        Detect and recognize license plates in image
        
        Args:
            image: Input image
            conf: Confidence threshold for detection
            fast_mode: If True, skip heavy fallback detection and OCR variants
            
        Returns:
            List of detections with recognized text
        """
        try:
            detections = self._run_detection_passes(image, conf, fast_mode=fast_mode)
            
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
                
                # Recognize text using multiple OCR variants (if not fast_mode)
                text, ocr_confidence = self._recognize_plate_text(roi, fast_mode=fast_mode)

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
