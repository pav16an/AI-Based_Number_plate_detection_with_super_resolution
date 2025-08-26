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

reader = easyocr.Reader(['en'])
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

def easy_ocr(frame, x1, y1, x2, y2):
    """Extract text from license plate region using EasyOCR"""
    try:
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return ""
        
        result = reader.readtext(cropped, detail=0)
        text = result[0] if result else ""
        
        # Clean up the text
        pattern = re.compile('[\\W]')
        text = pattern.sub('', text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.replace("粤", "")
        
        return str(text)
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

def process_image(image_path, confidence=0.45):
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
                    if label:
                        license_plates.add(label)
                    
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

def process_frame(frame_data, confidence=0.45):
    """Process video frame for license plate detection"""
    if model is None:
        return [], []
    
    try:
        # Convert frame data to numpy array
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return [], []
        
        original_frame = frame.copy()
        license_plates = set()
        detections = []
        
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
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else confidence
                    
                    # Extract text using OCR
                    label = easy_ocr(original_frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                        
                        # Add detection info
                        detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'label': label,
                            'confidence': conf
                        })
        
        # Save results if any plates detected
        if license_plates:
            start_time = end_time = datetime.now()
            save_to_database(license_plates, start_time, end_time)
            save_json_data(license_plates, start_time, end_time)
        
        return detections, list(license_plates)
        
    except Exception as e:
        print(f"Frame Processing Error: {e}")
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
    """Process webcam frame for live detection"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        confidence = float(request.form.get('confidence', 0.45))
        
        if frame_file:
            frame_data = frame_file.read()
            detections, license_plates = process_frame(frame_data, confidence)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'license_plates': license_plates
            })
        
        return jsonify({'error': 'Invalid frame data'}), 400
        
    except Exception as e:
        print(f"Webcam Frame Processing Error: {e}")
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