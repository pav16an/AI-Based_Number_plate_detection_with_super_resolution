# License Plate Detection System

A real-time license plate detection and recognition system using YOLOv10 and EasyOCR with a Flask web interface.

## Features

- **Real-time Detection**: Live webcam detection with optimized performance
- **High Accuracy**: Enhanced OCR with multiple preprocessing techniques
- **Web Interface**: User-friendly Flask web application
- **Database Storage**: SQLite database for storing detected plates
- **Multi-format Support**: Images and videos supported
- **Fast Processing**: Optimized for real-time performance (~200ms per frame)

## Performance

- **Success Rate**: 66.7% on test images
- **Processing Speed**: ~200ms average per frame
- **Real-time FPS**: 4.8 theoretical FPS
- **Accuracy**: High-quality OCR with confidence scoring

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd License-Plate-Detection-System
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the web interface**
Open your browser and go to `http://localhost:5000`

## Usage

### Web Interface
- **Home**: Upload images for detection
- **Live Detection**: Real-time webcam detection
- **Results**: View detection history and statistics

### Webcam Detection
- Adjustable confidence threshold (0.3-0.8)
- Variable detection intervals (500-3000ms)
- Real-time statistics and FPS counter
- Auto-detection toggle

### Standalone Script
```bash
python main.py
```
Choose from:
1. Image file detection
2. Video file detection  
3. Webcam detection

## Configuration

### Detection Settings
- **Confidence Threshold**: 0.5 (balanced speed/accuracy)
- **Detection Interval**: 800ms (webcam)
- **Image Processing**: 640px width for optimal speed
- **OCR Quality**: Enhanced preprocessing pipeline

### Database
- SQLite database: `licensePlatesDatabase.db`
- Automatic table creation
- Stores: plate text, timestamps, detection metadata

## API Endpoints

- `POST /upload` - Upload image for detection
- `POST /process_frame` - Process webcam frame
- `GET /api/plates` - Get all detected plates
- `DELETE /api/plates/<id>` - Delete specific plate

## File Structure

```
├── app.py              # Main Flask application
├── main.py             # Standalone detection script
├── config.py           # Configuration settings
├── sqldb.py            # Database utilities
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── index.html      # Home page
│   ├── webcam.html     # Live detection
│   └── results.html    # Results page
├── data/               # Sample images/videos
├── weights/            # Model weights
│   └── best.pt         # Trained YOLOv10 model
└── yolov10/           # YOLOv10 implementation
```

## Technical Details

### Model
- **Architecture**: YOLOv10 for object detection
- **OCR Engine**: EasyOCR for text recognition
- **Preprocessing**: Multiple enhancement techniques
- **Validation**: Smart license plate format validation

### Optimizations
- Frame resizing for faster processing
- Simplified OCR pipeline for speed
- Database operations skipped during live detection
- Optimized confidence thresholds

## Requirements

- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- Flask 2.3+
- EasyOCR 1.7+
- Ultralytics 8.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Troubleshooting

### Common Issues
- **Low accuracy**: Ensure good lighting and clear license plates
- **Slow processing**: Enable GPU acceleration if available
- **Camera access**: Check browser permissions for webcam
- **Model loading**: Verify `weights/best.pt` exists

### Performance Tips
- Use GPU for faster processing
- Adjust confidence threshold based on your needs
- Optimize detection intervals for your hardware
- Ensure adequate lighting for better OCR results