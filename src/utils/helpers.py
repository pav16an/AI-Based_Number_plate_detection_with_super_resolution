"""Helper utilities for license plate detection system."""

from typing import Any, Optional, Tuple
from datetime import datetime


def _import_cv2():
    """Import cv2 lazily so non-CV code paths can still boot."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for image processing features") from exc
    return cv2


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode image to base64 string
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string or None if error
    """
    import base64
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def draw_detections(image: Any, detections: list, class_names: list) -> Any:
    """
    Draw detection boxes on image
    
    Args:
        image: Input image
        detections: List of detections with bounding boxes
        class_names: List of class names
        
    Returns:
        Image with drawn detections
    """
    cv2 = _import_cv2()
    image_copy = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        confidence = detection.get('confidence', 0)
        class_id = detection.get('class_id', 0)
        class_name = class_names[class_id] if class_id < len(class_names) else 'Unknown'
        text = detection.get('text', '')
        
        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {text} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 5), 
                      (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image_copy, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image_copy


def extract_roi(image: Any, bbox: Tuple[int, int, int, int], 
                padding: int = 5) -> Optional[Any]:
    """
    Extract region of interest from image
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding around the bounding box
        
    Returns:
        Extracted ROI or None
    """
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        return roi
    except Exception as e:
        print(f"Error extracting ROI: {e}")
        return None


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_timestamp() -> str:
    """Get current timestamp as ISO format string"""
    return datetime.utcnow().isoformat()


def format_confidence(confidence: float, decimals: int = 2) -> str:
    """Format confidence value as percentage"""
    return f"{confidence * 100:.{decimals}f}%"
