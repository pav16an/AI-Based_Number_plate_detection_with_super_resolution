---
title: License Plate Detection
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app_gradio.py
pinned: false
license: mit
---

# License Plate Detection System

A real-time license plate detection and recognition system using YOLOv10 and EasyOCR.

## Features

- **YOLOv10 Detection**: Fast and accurate license plate localization
- **EasyOCR Recognition**: High-quality text extraction
- **Real-time Processing**: Optimized for speed (~200ms per image)
- **Web Interface**: User-friendly Gradio interface

## How to Use

1. Upload an image containing vehicles
2. Click "Detect License Plates"
3. View the results with bounding boxes and extracted text

## Model Performance

- **Success Rate**: 66.7% on test images
- **Processing Speed**: ~200ms per image
- **Supported Formats**: JPG, PNG, JPEG

## Technical Details

- **Detection Model**: Custom trained YOLOv10
- **OCR Engine**: EasyOCR with enhanced preprocessing
- **Image Processing**: OpenCV for optimization
- **Validation**: Smart license plate format checking

## Examples

The system works well with:
- Clear, well-lit license plates
- Various angles and distances
- Multiple plates in single image
- Different license plate formats