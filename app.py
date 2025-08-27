from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import json
import sqlite3
from datetime import datetime
import easyocr
import re
import math
from ultralytics import YOLO
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('json', exist_ok=True)

# Initialize YOLO model and EasyOCR
try:
    model = YOLO(os.path.join("weights", "best.pt"))
    print("Custom YOLOv10 model loaded successfully")
except Exception as e:
    print(f"Error loading custom model: {e}")
    try:
        model = YOLO("yolov8n.pt")
        print("Standard YOLO model loaded successfully")
    except Exception as e2:
        print(f"Error loading standard model: {e2}")
        model = None

# Initialize EasyOCR with optimized settings for license plates
try:
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
    print("EasyOCR initialized successfully")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    reader = None
className = ["License"]

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    
    # Create table with all required columns if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            license_plate TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def validate_license_plate(text):
    """Enhanced validation for license plate text"""
    if not text or len(text) < 3 or len(text) > 10:
        return False
    
    # Must contain at least one digit
    if not any(c.isdigit() for c in text):
        return False
    
    # Should not be all digits or all letters
    if text.isdigit() or text.isalpha():
        return False
    
    # Common OCR errors to reject
    invalid_patterns = ['III', '000', 'OOO', 'LLL', '111']
    if text in invalid_patterns:
        return False
    
    return True

def fast_ocr(frame, x1, y1, x2, y2):
    """Fast OCR optimized for webcam detection"""
    try:
        # Extract ROI
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 15 or roi.shape[1] < 30:
            return ""
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Simple but effective preprocessing
        # Resize if too small
        if gray.shape[0] < 32:
            scale = 32 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_CUBIC)
        
        # Basic enhancement
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Fast OCR
        results = reader.readtext(thresh, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=1)
        
        best_text = ""
        best_conf = 0
        
        for (bbox, text, conf) in results:
            if conf > best_conf and len(text) >= 3:
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if 3 <= len(clean_text) <= 10 and any(c.isdigit() for c in clean_text) and any(c.isalpha() for c in clean_text):
                    best_text = clean_text
                    best_conf = conf
        
        return best_text if best_conf > 0.6 else ""
        
    except Exception:
        return ""

def easy_ocr(frame, x1, y1, x2, y2):
    """Extract text from license plate region using EasyOCR with enhanced preprocessing"""
    try:
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract the license plate region
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return ""
        
        # Convert to grayscale
        if len(cropped.shape) == 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped
        
        # Multiple preprocessing approaches
        processed_images = []
        
        # 1. Original grayscale
        processed_images.append(("original", gray))
        
        # 2. Resize if too small
        height, width = gray.shape
        if height < 40:
            scale = 40 / height
            new_width = int(width * scale)
            resized = cv2.resize(gray, (new_width, 40), interpolation=cv2.INTER_CUBIC)
            processed_images.append(("resized", resized))
        
        # 3. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(("enhanced", enhanced))
        
        # 4. Gaussian blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        processed_images.append(("adaptive", adaptive))
        
        # 5. OTSU thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(("otsu", otsu))
        
        # 6. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        processed_images.append(("morph", morph))
        
        best_result = ""
        best_confidence = 0
        
        # Try OCR on each processed image
        for name, img in processed_images:
            try:
                # Method 1: EasyOCR with allowlist
                results = reader.readtext(img, 
                                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                                        width_ths=0.7,
                                        height_ths=0.7,
                                        detail=1)
                
                for (bbox, text, confidence) in results:
                    if confidence > best_confidence and len(text) >= 3:
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if validate_license_plate(clean_text):
                            best_result = clean_text
                            best_confidence = confidence
                            print(f"OCR ({name}): '{clean_text}' conf={confidence:.3f}")
                
                # Method 2: Fallback without allowlist
                if not best_result:
                    results = reader.readtext(img, detail=1)
                    for (bbox, text, confidence) in results:
                        if confidence > 0.4 and len(text) >= 3:
                            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                            if validate_license_plate(clean_text) and confidence > best_confidence:
                                best_result = clean_text
                                best_confidence = confidence
                                
            except Exception as ocr_error:
                continue
        
        # Final validation and scoring
        if best_result and best_confidence > 0.5:
            # Score the result
            score = 0
            if 6 <= len(best_result) <= 8:
                score += 3
            elif 4 <= len(best_result) <= 9:
                score += 2
            
            letters = sum(1 for c in best_result if c.isalpha())
            digits = sum(1 for c in best_result if c.isdigit())
            if 2 <= letters <= 4 and 2 <= digits <= 4:
                score += 2
            
            if score >= 2:  # Only return high-quality results
                print(f"Final OCR Result: '{best_result}' (confidence: {best_confidence:.2f}, score: {score})")
                return best_result
        
        return ""
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def save_to_database(license_plates, start_time, end_time):
    """Save license plates to database"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        current_time = datetime.now().isoformat()
        
        for plate in license_plates:
            cursor.execute('''
                INSERT INTO LicensePlates(start_time, end_time, license_plate, created_at)
                VALUES (?, ?, ?, ?)
            ''', (start_time.isoformat(), end_time.isoformat(), plate, current_time))
            print(f"Saved license plate to database: {plate}")
        
        conn.commit()
        conn.close()
        print(f"Successfully saved {len(license_plates)} license plates to database")
    except Exception as e:
        print(f"Database Error: {e}")
        import traceback
        traceback.print_exc()

def save_json_data(license_plates, start_time, end_time):
    """Save data to JSON files"""
    try:
        # Individual JSON file
        interval_data = {
            "Start Time": start_time.isoformat(),
            "End Time": end_time.isoformat(),
            "License Plate": list(license_plates)
        }
        
        interval_file_path = f"json/output_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(interval_file_path, 'w') as f:
            json.dump(interval_data, f, indent=2)
        
        # Cumulative JSON file
        cumulative_file_path = "json/LicensePlateData.json"
        if os.path.exists(cumulative_file_path):
            with open(cumulative_file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        existing_data.append(interval_data)
        
        with open(cumulative_file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
    except Exception as e:
        print(f"JSON Save Error: {e}")

def process_image(image_path, confidence=0.6):
    """Process image for license plate detection"""
    if model is None:
        return None, []
    
    try:
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            return None, []
        
        original_frame = frame.copy()
        license_plates = set()
        
        # Run YOLO detection
        results = model.predict(frame, conf=confidence)
        
        for result in results:
            # Handle different result formats
            if hasattr(result, 'boxes'):
                boxes = result.boxes
            else:
                boxes = result
                if hasattr(boxes, 'xyxy'):
                    boxes = [boxes]
            
            for box in boxes:
                if hasattr(box, 'xyxy'):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Extract text using OCR
                    label = easy_ocr(original_frame, x1, y1, x2, y2)
                    if label and len(label) >= 3:
                        license_plates.add(label)
                        print(f"Image detection - License plate: {label}")
                    
                    # Add text label
                    if label:
                        textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                        c2 = x1 + textSize[0], y1 - textSize[1] - 3
                        cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], 
                                  thickness=1, lineType=cv2.LINE_AA)
        
        # Save results
        start_time = end_time = datetime.now()
        if license_plates:
            save_to_database(license_plates, start_time, end_time)
            save_json_data(license_plates, start_time, end_time)
        
        return frame, list(license_plates)
        
    except Exception as e:
        print(f"Processing Error: {e}")
        return None, []

def process_frame(frame_data, confidence=0.5):
    """Fast and accurate frame processing for webcam detection"""
    if model is None:
        return [], []
    
    try:
        # Convert frame data to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return [], []
        
        # Resize frame for faster processing while maintaining quality
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_w, new_h = 640, int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            scale_factor = w / 640
        else:
            frame_resized = frame
            scale_factor = 1.0
        
        license_plates = []
        detections = []
        
        # Run YOLO detection on resized frame for speed
        results = model.predict(frame_resized, conf=confidence, verbose=False, imgsz=640)
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                # Process up to 3 best detections for speed
                num_boxes = min(len(boxes), 3)
                
                for i in range(num_boxes):
                    try:
                        x1, y1, x2, y2 = boxes.xyxy[i]
                        conf = float(boxes.conf[i])
                        
                        # Scale coordinates back to original size
                        x1, y1, x2, y2 = int(x1 * scale_factor), int(y1 * scale_factor), int(x2 * scale_factor), int(y2 * scale_factor)
                        
                        # Quick validation
                        if x2 <= x1 or y2 <= y1 or (x2-x1) < 30 or (y2-y1) < 15:
                            continue
                        
                        # Fast OCR with simplified preprocessing
                        label = fast_ocr(frame, x1, y1, x2, y2)
                        
                        if label:
                            license_plates.append(label)
                            detections.append({
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'label': label,
                                'confidence': conf
                            })
                        
                    except Exception:
                        continue
        
        # Only save if we have valid plates
        if license_plates:
            unique_plates = list(dict.fromkeys(license_plates))
            # Skip database save for webcam to improve speed
            print(f"Detected: {unique_plates}")
        
        return detections, license_plates
        
    except Exception as e:
        print(f"Frame error: {e}")
        return [], []

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/results')
def results():
    """Results page"""
    return render_template('results.html')

@app.route('/webcam')
def webcam():
    """Webcam live detection page"""
    return render_template('webcam.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            processed_frame, license_plates = process_image(filepath)
            
            if processed_frame is not None:
                # Convert processed image to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Save processed image
                processed_filename = f"processed_{filename}"
                processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_filepath, processed_frame)
                
                return jsonify({
                    'success': True,
                    'image': img_base64,
                    'license_plates': license_plates,
                    'filename': processed_filename
                })
            else:
                return jsonify({'error': 'Failed to process image'}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates')
def get_plates():
    """API endpoint to get all detected license plates"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # First check if table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='LicensePlates'")
        if not cursor.fetchone():
            conn.close()
            return jsonify([])
        
        # Get all records with proper ordering
        cursor.execute('''
            SELECT id, license_plate, start_time, end_time, 
                   COALESCE(created_at, start_time, datetime('now')) as created_at 
            FROM LicensePlates 
            ORDER BY id DESC 
            LIMIT 100
        ''')
        
        plates = cursor.fetchall()
        conn.close()
        
        result = []
        for plate in plates:
            result.append({
                'id': plate[0],
                'license_plate': plate[1] if plate[1] else 'Unknown',
                'start_time': plate[2] if plate[2] else '',
                'end_time': plate[3] if plate[3] else '',
                'created_at': plate[4] if plate[4] else plate[2]
            })
        
        print(f"Retrieved {len(result)} plates from database")
        if len(result) == 0:
            print("No data found in database")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error fetching plates: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_webcam_frame():
    """Process webcam frame for live detection with enhanced multi-plate support"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        confidence = float(request.form.get('confidence', 0.5))  # Balanced for speed and accuracy
        
        # Validate confidence range
        confidence = max(0.1, min(0.9, confidence))
        
        if frame_file:
            frame_data = frame_file.read()
            
            # Check if frame data is valid
            if len(frame_data) == 0:
                return jsonify({'error': 'Empty frame data'}), 400
            
            detections, license_plates = process_frame(frame_data, confidence)
            
            # Log detection results
            if license_plates:
                print(f"Webcam processed: {len(detections)} detections, {len(license_plates)} plates: {license_plates}")
            
            return jsonify({
                'success': True,
                'detections': detections,
                'license_plates': license_plates,
                'frame_size': len(frame_data),
                'confidence_used': confidence
            })
        
        return jsonify({'error': 'Invalid frame data'}), 400
        
    except ValueError as ve:
        print(f"Webcam Frame Processing Value Error: {ve}")
        return jsonify({'error': f'Invalid confidence value: {ve}'}), 400
    except Exception as e:
        print(f"Webcam Frame Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test_db')
def test_database():
    """Test database functionality by adding sample data"""
    try:
        # Add some test data
        test_plates = ['ABC123', 'XYZ789', 'TEST001']
        start_time = end_time = datetime.now()
        
        save_to_database(test_plates, start_time, end_time)
        
        # Query the data back
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM LicensePlates ORDER BY id DESC LIMIT 10')
        results = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Added {len(test_plates)} test plates',
            'recent_data': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_db')
def check_database():
    """Check database structure and content"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # Check table structure
        cursor.execute("PRAGMA table_info(LicensePlates)")
        columns = cursor.fetchall()
        
        # Check data count
        cursor.execute('SELECT COUNT(*) FROM LicensePlates')
        count = cursor.fetchone()[0]
        
        # Get recent data
        cursor.execute('SELECT * FROM LicensePlates ORDER BY id DESC LIMIT 5')
        recent_data = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'table_structure': columns,
            'total_records': count,
            'recent_data': recent_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates/<int:plate_id>', methods=['DELETE'])
def delete_plate(plate_id):
    """Delete a single license plate record"""
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute('SELECT id FROM LicensePlates WHERE id = ?', (plate_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Record not found'}), 404
        
        # Delete the record
        cursor.execute('DELETE FROM LicensePlates WHERE id = ?', (plate_id,))
        conn.commit()
        conn.close()
        
        print(f"Deleted license plate record with ID: {plate_id}")
        return jsonify({'success': True, 'message': 'Record deleted successfully'})
        
    except Exception as e:
        print(f"Error deleting plate: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates/bulk-delete', methods=['POST'])
def bulk_delete_plates():
    """Delete multiple license plate records"""
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            return jsonify({'error': 'No IDs provided'}), 400
        
        ids = data['ids']
        if not ids:
            return jsonify({'error': 'Empty ID list'}), 400
        
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        # Convert IDs to integers and create placeholders
        try:
            ids = [int(id_val) for id_val in ids]
        except ValueError:
            conn.close()
            return jsonify({'error': 'Invalid ID format'}), 400
        
        placeholders = ','.join('?' * len(ids))
        
        # Check how many records exist
        cursor.execute(f'SELECT COUNT(*) FROM LicensePlates WHERE id IN ({placeholders})', ids)
        existing_count = cursor.fetchone()[0]
        
        # Delete the records
        cursor.execute(f'DELETE FROM LicensePlates WHERE id IN ({placeholders})', ids)
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"Bulk deleted {deleted_count} license plate records")
        return jsonify({
            'success': True, 
            'message': f'Successfully deleted {deleted_count} records',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        print(f"Error bulk deleting plates: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    init_database()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
else:
    init_database()
    
# For Render deployment
if 'RENDER' in os.environ:
    init_database()