import gradio as gr
import cv2
import numpy as np
import os
import re
from ultralytics import YOLO
import easyocr
from PIL import Image

# Initialize model and OCR
try:
    model = YOLO("weights/best.pt")
    print("✓ Model loaded successfully")
except:
    model = None
    print("✗ Model not found")

try:
    reader = easyocr.Reader(['en'], gpu=False)
    print("✓ EasyOCR initialized")
except:
    reader = None
    print("✗ EasyOCR failed")

def validate_license_plate(text):
    if not text or len(text) < 3 or len(text) > 10:
        return False
    if not any(c.isdigit() for c in text):
        return False
    if not any(c.isalpha() for c in text):
        return False
    return True

def fast_ocr(frame, x1, y1, x2, y2):
    try:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return ""
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        if gray.shape[0] < 32:
            scale = 32 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        results = reader.readtext(thresh, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=1)
        
        best_text = ""
        best_conf = 0
        
        for (bbox, text, conf) in results:
            if conf > best_conf and len(text) >= 3:
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if validate_license_plate(clean_text):
                    best_text = clean_text
                    best_conf = conf
        
        return best_text if best_conf > 0.6 else ""
    except:
        return ""

def detect_license_plates(image):
    if model is None or reader is None:
        return image, "❌ Model or OCR not loaded"
    
    try:
        # Convert PIL to OpenCV
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize for faster processing
        h, w = image.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w, new_h = 640, int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
            scale_factor = w / 640
        else:
            image_resized = image
            scale_factor = 1.0
        
        # Run detection
        results = model.predict(image_resized, conf=0.5, verbose=False)
        
        detected_plates = []
        annotated_image = image.copy()
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(min(len(boxes), 3)):  # Max 3 detections
                    x1, y1, x2, y2 = boxes.xyxy[i]
                    conf = float(boxes.conf[i])
                    
                    # Scale back to original size
                    x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                    
                    # OCR
                    plate_text = fast_ocr(image, x1, y1, x2, y2)
                    
                    if plate_text:
                        detected_plates.append(plate_text)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add text
                        label = f"{plate_text} ({conf:.2f})"
                        cv2.putText(annotated_image, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if detected_plates:
            result_text = f"✅ Detected: {', '.join(detected_plates)}"
        else:
            result_text = "❌ No license plates detected"
        
        return annotated_image, result_text
        
    except Exception as e:
        return image, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="License Plate Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 License Plate Detection System")
    gr.Markdown("Upload an image to detect and recognize license plates using YOLOv10 + EasyOCR")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            detect_btn = gr.Button("🔍 Detect License Plates", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results")
            output_text = gr.Textbox(label="Detected Plates", lines=3)
    
    detect_btn.click(
        fn=detect_license_plates,
        inputs=input_image,
        outputs=[output_image, output_text]
    )
    
    gr.Examples(
        examples=[
            ["data/carImage2.png"],
            ["data/2642543461_4f83ac5e54_z_jpg.rf.976c7fe63ba347eac086fd6e06c4c556.jpg"]
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch()