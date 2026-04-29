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


def _register_ultralytics_compatibility_shims() -> None:
    """Register compatibility shims for checkpoints trained with other Ultralytics builds."""
    import torch
    import torch.nn as nn
    from ultralytics.nn import tasks as ultralytics_tasks
    from ultralytics.nn.modules import block as ultralytics_block

    # ── 1. YOLOv10DetectionModel alias ───────────────────────────────────────
    if not hasattr(ultralytics_tasks, 'YOLOv10DetectionModel') and hasattr(ultralytics_tasks, 'DetectionModel'):
        ultralytics_tasks.YOLOv10DetectionModel = ultralytics_tasks.DetectionModel
        logger.info("Registered YOLOv10DetectionModel compatibility alias")

    Conv = ultralytics_block.Conv

    # ── 2. SCDown ─────────────────────────────────────────────────────────────
    if not hasattr(ultralytics_block, 'SCDown'):
        class SCDown(nn.Module):
            """Spatial-Channel Downsample (YOLOv10)."""
            def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, *args, **kwargs):
                super().__init__()
                self.cv1 = Conv(c1, c2, 1, 1)
                self.cv2 = Conv(c2, c2, k, s, g=c2, act=False)

            def forward(self, x):
                return self.cv2(self.cv1(x))

        ultralytics_block.SCDown = SCDown
        logger.info("Registered SCDown compatibility shim")

    # ── 3. Attention (needed by PSA) ──────────────────────────────────────────
    if not hasattr(ultralytics_block, 'Attention'):
        class Attention(nn.Module):
            """Multi-head self-attention used inside PSA (YOLOv10)."""
            def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = dim // num_heads
                self.key_dim = int(self.head_dim * attn_ratio)
                self.scale = self.key_dim ** -0.5
                nh_kd = self.key_dim * num_heads
                h = dim + nh_kd * 2
                self.qkv = Conv(dim, h, 1, act=False)
                self.proj = Conv(dim, dim, 1, act=False)
                self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

            def forward(self, x):
                B, C, H, W = x.shape
                N = H * W
                qkv = self.qkv(x)
                q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
                    [self.key_dim, self.key_dim, self.head_dim], dim=2
                )
                attn = (q.transpose(-2, -1) @ k) * self.scale
                attn = attn.softmax(dim=-1)
                x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
                return self.proj(x)

        ultralytics_block.Attention = Attention
        logger.info("Registered Attention compatibility shim")

    # ── 4. PSA (Partial Self-Attention) ───────────────────────────────────────
    if not hasattr(ultralytics_block, 'PSA'):
        _Attention = ultralytics_block.Attention

        class PSA(nn.Module):
            """Partial Self-Attention block (YOLOv10)."""
            def __init__(self, c1: int, c2: int, e: float = 0.5):
                super().__init__()
                assert c1 == c2
                self.c = int(c1 * e)
                self.cv1 = Conv(c1, 2 * self.c, 1, 1)
                self.cv2 = Conv(2 * self.c, c1, 1)
                self.attn = _Attention(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64))
                self.ffn = nn.Sequential(
                    Conv(self.c, self.c * 2, 1),
                    Conv(self.c * 2, self.c, 1, act=False),
                )

            def forward(self, x):
                a, b = self.cv1(x).split((self.c, self.c), dim=1)
                b = b + self.attn(b)
                b = b + self.ffn(b)
                return self.cv2(torch.cat((a, b), 1))

        ultralytics_block.PSA = PSA
        logger.info("Registered PSA compatibility shim")

    # ── 5. CIB (Compact Inverted Block) ───────────────────────────────────────
    if not hasattr(ultralytics_block, 'CIB'):
        class CIB(nn.Module):
            """Compact Inverted Block (YOLOv10)."""
            def __init__(self, c1: int, c2: int, shortcut: bool = True,
                         e: float = 0.5, lk: bool = False):
                super().__init__()
                c_ = int(c2 * e)
                self.cv1 = nn.Sequential(
                    Conv(c1, c1, 3, g=c1),
                    Conv(c1, 2 * c_, 1),
                    Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
                    Conv(2 * c_, c2, 1),
                    Conv(c2, c2, 3, g=c2),
                )
                self.add = shortcut and c1 == c2

            def forward(self, x):
                return x + self.cv1(x) if self.add else self.cv1(x)

        ultralytics_block.CIB = CIB
        logger.info("Registered CIB compatibility shim")

    # ── 6. C2fCIB ─────────────────────────────────────────────────────────────
    if not hasattr(ultralytics_block, 'C2fCIB') and hasattr(ultralytics_block, 'C2f'):
        _C2f = ultralytics_block.C2f
        _CIB = ultralytics_block.CIB

        class C2fCIB(_C2f):
            """C2f with CIB bottleneck (YOLOv10)."""
            def __init__(self, c1: int, c2: int, n: int = 1,
                         shortcut: bool = False, lk: bool = False,
                         g: int = 1, e: float = 0.5):
                super().__init__(c1, c2, n, shortcut, g, e)
                self.m = nn.ModuleList(
                    _CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n)
                )

        ultralytics_block.C2fCIB = C2fCIB
        logger.info("Registered C2fCIB compatibility shim")

    # ── 7. RepVGGDW ───────────────────────────────────────────────────────────
    if not hasattr(ultralytics_block, 'RepVGGDW'):
        class RepVGGDW(nn.Module):
            """Reparameterizable VGG-style depthwise conv block (YOLOv10)."""
            def __init__(self, ed: int):
                super().__init__()
                self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
                self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
                self.dim = ed
                self.act = nn.SiLU()

            def forward(self, x):
                return self.act(self.conv(x) + self.conv1(x))

            def forward_fuse(self, x):
                return self.act(self.conv(x))

        ultralytics_block.RepVGGDW = RepVGGDW
        logger.info("Registered RepVGGDW compatibility shim")

    # ── 8. v10Detect head ─────────────────────────────────────────────────────
    try:
        from ultralytics.nn.modules import head as ultralytics_head
        if not hasattr(ultralytics_head, 'v10Detect') and hasattr(ultralytics_head, 'Detect'):
            _Detect = ultralytics_head.Detect

            class v10Detect(_Detect):
                """YOLOv10 dual-assignment detection head compatibility shim."""
                max_det: int = 300

                def __init__(self, nc: int = 80, ch: tuple = ()):
                    super().__init__(nc, ch)
                    c2 = max(ch[0] // 4, self.reg_max * 4, 16) if ch else 16
                    c3 = max(ch[0], min(self.nc, 100)) if ch else 100
                    self.one2one_cv2 = nn.ModuleList(
                        nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                                      nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
                    )
                    self.one2one_cv3 = nn.ModuleList(
                        nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                      nn.Conv2d(c3, self.nc, 1)) for x in ch
                    )

            ultralytics_head.v10Detect = v10Detect
            # Also expose via tasks module so pickle finds it
            ultralytics_tasks.v10Detect = v10Detect
            logger.info("Registered v10Detect compatibility shim")
    except Exception as _shim_exc:
        logger.debug("Could not register v10Detect shim: %s", _shim_exc)

    # ── 9. v10DetectLoss (stored in checkpoint's loss attribute) ──────────────
    try:
        from ultralytics.utils import loss as ultralytics_loss
        if not hasattr(ultralytics_loss, 'v10DetectLoss'):
            _E2EDetectLoss = getattr(ultralytics_loss, 'E2EDetectLoss', None)
            _v8DetectionLoss = getattr(ultralytics_loss, 'v8DetectionLoss', None)
            _base = _E2EDetectLoss or _v8DetectionLoss
            if _base is not None:
                class v10DetectLoss(_base):
                    pass
            else:
                import torch.nn as _nn
                class v10DetectLoss(_nn.Module):
                    def __init__(self, *args, **kwargs):
                        super().__init__()
                    def forward(self, *args, **kwargs):
                        raise NotImplementedError()
            ultralytics_loss.v10DetectLoss = v10DetectLoss
            logger.info('Registered v10DetectLoss compatibility shim')
    except Exception as _shim_exc:
        logger.debug('Could not register v10DetectLoss shim: %s', _shim_exc)

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
            _register_ultralytics_compatibility_shims()
            
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
                    self.model.fuse()
                    logger.debug("Fused YOLO model layers for optimized inference")
                except Exception:
                    logger.debug("YOLO model fusion not available")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def predict(self, image: Any, conf: Optional[float] = None, iou: float = 0.45,
                max_det: int = 50, fast_mode: bool = False) -> List[Dict]:
        """
        Detect license plates in image
        
        Args:
            image: Input image
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold for non-max suppression
            max_det: Maximum number of detections to return
            fast_mode: If True, use smaller imgsz (640) for much faster inference
            
        Returns:
            List of detections with bounding boxes and confidence
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            confidence = conf or self.confidence
            # Use 640 for real-time/fast-mode (≈3× faster than 960+).
            # Use 960 for high-quality single-image uploads only.
            if fast_mode:
                imgsz = 640
            else:
                image_height, image_width = image.shape[:2]
                max_side = max(image_height, image_width)
                imgsz = min(1280, max(640, ((max_side + 31) // 32) * 32))

            results = self.model(
                image,
                conf=confidence,
                iou=iou,
                max_det=max_det,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().astype(int),
                        'confidence': float(box.conf[0].cpu()),
                        'class_id': int(box.cls[0].cpu())
                    }
                    detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} license plates (imgsz={imgsz})")
            return detections
        except Exception as e:
            logger.error(f"Error during detection: {e}")
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

        # Upscale the whole image for small plate recovery.
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
