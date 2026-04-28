# API Documentation

## License Plate Detection System - REST API v1

### Base URL
```
http://localhost:5000/api/v1
```

### Response Format
All responses are in JSON format with the following structure:

**Success Response:**
```json
{
  "data": {...},
  "success": true,
  "status_code": 200
}
```

**Error Response:**
```json
{
  "message": "Error description",
  "success": false,
  "status_code": 400
}
```

---

## Endpoints

### 1. Health Check
Check if the API is operational.

**Endpoint:**
```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "v1",
  "success": true
}
```

---

### 2. Detect License Plate in Image
Upload an image and detect license plates.

**Endpoint:**
```
POST /detect/image
```

**Content-Type:**
```
multipart/form-data
```

**Request:**
- `file`: Image file (PNG, JPG, JPEG, GIF)

**cURL Example:**
```bash
curl -X POST \
  http://localhost:5000/api/v1/detect/image \
  -F "file=@image.jpg"
```

**Response (200 OK):**
```json
{
  "detections": [
    {
      "bbox": [100, 150, 250, 200],
      "confidence": 0.95,
      "text": "ABC123",
      "ocr_confidence": 0.87
    }
  ],
  "count": 1,
  "image_path": "uploads/image_20240101_120000.jpg",
  "success": true
}
```

**Error Responses:**
- 400: No file provided
- 422: Invalid file type
- 500: Processing error

---

### 3. Real-time Webcam Detection
Process a single frame from webcam stream.

**Endpoint:**
```
POST /detect/webcam?confidence=0.6
```

**Content-Type:**
```
application/json
```

**Query Parameters:**
- `confidence` (optional): Detection threshold (0.0-1.0, default: 0.6)

**Request Body:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

**Response (200 OK):**
```json
{
  "detections": [
    {
      "bbox": [120, 160, 260, 210],
      "confidence": 0.92,
      "text": "XYZ789",
      "ocr_confidence": 0.89
    }
  ],
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA...",
  "count": 1,
  "success": true
}
```

**Python Example:**
```python
import requests
import base64
import cv2

# Capture frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Encode to base64
_, buffer = cv2.imencode('.jpg', frame)
frame_base64 = base64.b64encode(buffer).decode()

# Send to API
response = requests.post(
    'http://localhost:5000/api/v1/detect/webcam?confidence=0.6',
    json={'frame': f'data:image/jpeg;base64,{frame_base64}'}
)

print(response.json())
```

---

### 4. Get All Detections
Retrieve all stored detections with pagination.

**Endpoint:**
```
GET /detections?page=1&per_page=50
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Results per page (default: 50, max: 100)

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/detections?page=1&per_page=50"
```

**Response (200 OK):**
```json
{
  "detections": [
    {
      "id": 1,
      "license_plate": "ABC123",
      "confidence": 0.95,
      "source": "image",
      "timestamp": "2024-01-01T12:00:00",
      "image_path": "uploads/image.jpg",
      "metadata": {},
      "status": "success",
      "created_at": "2024-01-01T12:00:05"
    }
  ],
  "page": 1,
  "per_page": 50,
  "count": 1,
  "success": true
}
```

---

### 5. Get Detection by ID
Retrieve a specific detection record.

**Endpoint:**
```
GET /detections/<detection_id>
```

**URL Parameters:**
- `detection_id`: ID of the detection

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/detections/1"
```

**Response (200 OK):**
```json
{
  "detection": {
    "id": 1,
    "license_plate": "ABC123",
    "confidence": 0.95,
    "source": "image",
    "timestamp": "2024-01-01T12:00:00",
    "image_path": "uploads/image.jpg",
    "metadata": {},
    "status": "success",
    "created_at": "2024-01-01T12:00:05"
  },
  "success": true
}
```

**Error Response (404 Not Found):**
```json
{
  "message": "Detection 999 not found",
  "success": false
}
```

---

### 6. Search Detections
Search for detections by license plate.

**Endpoint:**
```
GET /search?plate=ABC
```

**Query Parameters:**
- `plate`: License plate text to search (required, minimum 2 characters)

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/search?plate=ABC"
```

**Response (200 OK):**
```json
{
  "query": "ABC",
  "detections": [
    {
      "id": 1,
      "license_plate": "ABC123",
      "confidence": 0.95,
      "source": "image",
      "timestamp": "2024-01-01T12:00:00",
      "image_path": "uploads/image.jpg",
      "metadata": {},
      "status": "success",
      "created_at": "2024-01-01T12:00:05"
    }
  ],
  "count": 1,
  "success": true
}
```

---

### 7. Get Statistics
Retrieve detection statistics for a specific date.

**Endpoint:**
```
GET /statistics?date=2024-01-01
```

**Query Parameters:**
- `date` (optional): Date in YYYY-MM-DD format (default: today)

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/statistics?date=2024-01-01"
```

**Response (200 OK):**
```json
{
  "date": "2024-01-01",
  "statistics": {
    "id": 1,
    "date": "2024-01-01",
    "total_detections": 150,
    "successful_detections": 100,
    "failed_detections": 50,
    "average_confidence": 0.87,
    "unique_plates": 45,
    "created_at": "2024-01-02T00:00:00"
  },
  "success": true
}
```

---

### 8. Download Uploaded File
Download an uploaded image or video file.

**Endpoint:**
```
GET /uploads/<filename>
```

**URL Parameters:**
- `filename`: Name of the file to download

**cURL Example:**
```bash
curl "http://localhost:5000/api/v1/uploads/image_20240101_120000.jpg" \
  -o downloaded_image.jpg
```

**Response (200 OK):**
- File content with appropriate Content-Type header

**Error Response (404 Not Found):**
```json
{
  "message": "File not found",
  "success": false
}
```

---

## Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 200 | OK | Success |
| 400 | Bad Request | Invalid input or missing parameters |
| 404 | Not Found | Resource not found |
| 405 | Method Not Allowed | HTTP method not allowed |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error |

---

## Rate Limiting

The API implements rate limiting to prevent abuse:
- **Default**: 100 requests per 1 hour per IP
- Can be configured in `.env`: `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_PERIOD`

Response headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1704114000
```

---

## Authentication

Currently, no authentication is required. For production use, implement:
- API Key authentication
- JWT tokens
- OAuth 2.0

---

## Examples

### Python Integration
```python
import requests
import json

BASE_URL = 'http://localhost:5000/api/v1'

# Health check
response = requests.get(f'{BASE_URL}/health')
print(response.json())

# Detect image
with open('license_plate.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(f'{BASE_URL}/detect/image', files=files)
    print(response.json())

# Get all detections
response = requests.get(f'{BASE_URL}/detections?page=1&per_page=50')
print(response.json())

# Search detections
response = requests.get(f'{BASE_URL}/search?plate=ABC')
print(response.json())
```

### JavaScript/Node.js Integration
```javascript
const API_BASE_URL = 'http://localhost:5000/api/v1';

// Health check
fetch(`${API_BASE_URL}/health`)
  .then(res => res.json())
  .then(data => console.log(data));

// Detect image
const formData = new FormData();
formData.append('file', imageFile);

fetch(`${API_BASE_URL}/detect/image`, {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data));

// Get detections
fetch(`${API_BASE_URL}/detections?page=1&per_page=50`)
  .then(res => res.json())
  .then(data => console.log(data));
```

---

## Version History

### v1.0.0 (Current)
- Initial release
- License plate detection endpoint
- Webcam detection endpoint
- Database storage
- Statistics tracking

---

**Last Updated**: 2024
**API Version**: 1.0.0
