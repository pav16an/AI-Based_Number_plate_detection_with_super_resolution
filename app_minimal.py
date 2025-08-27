from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import json
import sqlite3
from datetime import datetime
import re

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

def simple_ocr(image_region):
    # Placeholder OCR - returns demo text
    return "DEMO123"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Simple demo response
            return jsonify({
                'success': True,
                'license_plates': ['DEMO123'],
                'image': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k='
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_webcam_frame():
    try:
        # Demo response for webcam
        return jsonify({
            'success': True,
            'detections': [{'x1': 100, 'y1': 100, 'x2': 200, 'y2': 150, 'label': 'DEMO123', 'confidence': 0.85}],
            'license_plates': ['DEMO123']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plates')
def get_plates():
    try:
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, license_plate, start_time, end_time, created_at FROM LicensePlates ORDER BY id DESC LIMIT 10')
        plates = cursor.fetchall()
        conn.close()
        
        result = []
        for plate in plates:
            result.append({
                'id': plate[0],
                'license_plate': plate[1] or 'DEMO123',
                'start_time': plate[2] or '',
                'end_time': plate[3] or '',
                'created_at': plate[4] or datetime.now().isoformat()
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify([])

if __name__ == '__main__':
    init_database()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
else:
    init_database()

if 'RENDER' in os.environ:
    init_database()