from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import uuid
import base64
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('json', exist_ok=True)

def init_database():
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
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

def save_to_database(license_plates, start_time, end_time):
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        current_time = datetime.now().isoformat()
        
        for plate in license_plates:
            cursor.execute('''
                INSERT INTO LicensePlates(start_time, end_time, license_plate, created_at)
                VALUES (?, ?, ?, ?)
            ''', (start_time.isoformat(), end_time.isoformat(), plate, current_time))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Mock detection for demo
            mock_plates = ['ABC123', 'XYZ789']
            start_time = end_time = datetime.now()
            save_to_database(mock_plates, start_time, end_time)
            
            # Return mock response
            with open(filepath, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': img_base64,
                'license_plates': mock_plates,
                'filename': filename
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_webcam_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        frame_file = request.files['frame']
        confidence = float(request.form.get('confidence', 0.45))
        
        if frame_file:
            # Mock detection for demo
            import random
            mock_plates = []
            mock_detections = []
            
            # More frequent mock detections
            if random.random() > 0.2:  # 80% chance of detection
                plate_options = ['ABC123', 'XYZ789', 'DEF456', 'GHI789', 'JKL012', 'MNO345', 'PQR678']
                detected_plate = random.choice(plate_options)
                mock_plates.append(detected_plate)
                
                # Mock bounding box coordinates (more realistic)
                x1 = random.randint(100, 300)
                y1 = random.randint(100, 200)
                w = random.randint(120, 200)
                h = random.randint(40, 80)
                
                mock_detections.append({
                    'x1': x1,
                    'y1': y1, 
                    'x2': x1 + w,
                    'y2': y1 + h,
                    'label': detected_plate,
                    'confidence': round(confidence + random.uniform(-0.1, 0.2), 2)
                })
                
                # Save to database
                start_time = end_time = datetime.now()
                save_to_database(mock_plates, start_time, end_time)
            
            return jsonify({
                'success': True,
                'detections': mock_detections,
                'license_plates': mock_plates
            })
        
        return jsonify({'error': 'Invalid frame data'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates')
def get_plates():
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='LicensePlates'")
        if not cursor.fetchone():
            conn.close()
            return jsonify([])
        
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
                'license_plate': plate[1],
                'start_time': plate[2],
                'end_time': plate[3],
                'created_at': plate[4]
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates/<int:plate_id>', methods=['DELETE'])
def delete_plate(plate_id):
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM LicensePlates WHERE id = ?', (plate_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates/bulk-delete', methods=['POST'])
def bulk_delete_plates():
    try:
        data = request.get_json()
        ids = data['ids']
        
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(ids))
        cursor.execute(f'DELETE FROM LicensePlates WHERE id IN ({placeholders})', ids)
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'deleted_count': len(ids)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_database()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
else:
    init_database()