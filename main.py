#Import All the Required Libraries
import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
import easyocr
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Input selection menu
print("Select input source:")
print("1. Image file")
print("2. Video file")
print("3. Webcam")
choice = input("Enter your choice (1-3): ")

# Create Video Capture Object based on choice
if choice == '1':
    image_path = input("Enter image file path: ")
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        cap = None
    else:
        print(f"Error: File not found '{image_path}'")
        exit(1)
elif choice == '2':
    video_path = input("Enter video file path: ")
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print(f"Error: File not found '{video_path}'")
        exit(1)
elif choice == '3':
    cap = cv2.VideoCapture(0)  # Webcam
    print("Webcam started. Press 'q' to stop processing.")
else:
    print("Error: Invalid choice")
    exit(1)
#Initialize the YOLO Model
try:
    # Try to load the custom YOLOv10 model
    model = YOLO(os.path.join(os.path.dirname(__file__), "weights", "best.pt"))
    print("Custom YOLOv10 model loaded successfully")
except Exception as e:
    print(f"Error loading custom model: {e}")
    print("Falling back to standard YOLO model...")
    try:
        # Fallback to a standard YOLO model
        model = YOLO("yolov8n.pt")  # Use a standard YOLOv8 nano model
        print("Standard YOLO model loaded successfully")
    except Exception as e2:
        print(f"Error loading standard model: {e2}")
        print("Please check your model files and dependencies")
        exit(1)
#Initialize the frame count
count = 0
#Class Names
className = ["License"]
# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def easy_ocr(frame, x1, y1, x2, y2):
    # Ensure coordinates are within bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return ""
    
    # Convert to grayscale
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped
    
    # Resize for better OCR (minimum height 50px)
    height, width = gray.shape
    if height < 50:
        scale = 50 / height
        new_width = int(width * scale)
        gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
    
    # Image enhancement
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Try OCR with enhanced image
    try:
        results = reader.readtext(thresh, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=1)
        best_text = ""
        best_conf = 0
        
        for (bbox, text, conf) in results:
            if conf > best_conf and len(text) >= 3:
                best_text = text
                best_conf = conf
        
        if not best_text or best_conf < 0.6:
            # Fallback to original image
            results = reader.readtext(gray, detail=0)
            best_text = results[0] if results else ""
        
        # Clean and validate text
        if best_text:
            clean_text = re.sub(r'[^A-Z0-9]', '', best_text.upper())
            # Must be 3-10 chars and contain at least one digit
            if 3 <= len(clean_text) <= 10 and any(c.isdigit() for c in clean_text):
                return clean_text
        
        return ""
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""



def save_json(license_plates, startTime, endTime):
    #Generate individual JSON files for each 20-second interval
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent = 2)

    #Cummulative JSON File
    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    #Add new intervaal data to cummulative data
    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent = 2)

    #Save data to SQL database
    save_to_database(license_plates, startTime, endTime)



def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()



startTime = datetime.now()
license_plates = set()


# Process image if source is an image file
if cap is None:
    frame = image.copy()
    results = model.predict(frame, conf=0.6)
    for result in results:
        # Handle both old and new result formats
        if hasattr(result, 'boxes'):
            boxes = result.boxes
        else:
            # For newer YOLOv10 format
            boxes = result
            if hasattr(boxes, 'xyxy'):
                boxes = [boxes]  # Convert to list format
        
        for box in boxes:
            if hasattr(box, 'xyxy'):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = classNameInt
                conf = math.ceil(box.conf[0]*100)/100
                label = easy_ocr(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
    
    # Save results for image
    endTime = datetime.now()
    save_json(license_plates, startTime, endTime)
    cv2.imshow("License Plate Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Video/webcam processing
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Empty frame received")
            continue
            
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf=0.6)
        for result in results:
            # Handle both old and new result formats
            if hasattr(result, 'boxes'):
                boxes = result.boxes
            else:
                # For newer YOLOv10 format
                boxes = result
                if hasattr(boxes, 'xyxy'):
                    boxes = [boxes]  # Convert to list format
            
            for box in boxes:
                if hasattr(box, 'xyxy'):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    classNameInt = int(box.cls[0])
                    clsName = className[classNameInt]
                    conf = math.ceil(box.conf[0]*100)/100
                    #label = f'{clsName}:{conf}'
                    label = easy_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()
        if frame.size > 0:
            cv2.imshow("Video", frame)
        else:
            print("Warning: Skipping display of empty frame")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping webcam processing...")
            break

    cap.release()
cv2.destroyAllWindows()
