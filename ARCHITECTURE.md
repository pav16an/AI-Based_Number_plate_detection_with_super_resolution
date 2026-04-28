# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│  (Web Browser, Mobile App, API Client, Webcam Stream)               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    HTTP/REST│ (JSON)
                             │
┌─────────────────────────────▼────────────────────────────────────────┐
│                      FLASK APPLICATION                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    src/api/routes_v1.py                      │  │
│  │  • /health            • /detect/image                        │  │
│  │  • /detect/webcam     • /detections                          │  │
│  │  • /search            • /statistics                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            Error Handlers & Middleware                       │  │
│  │  • API Error handling  • Logging  • Validation              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   SERVICES LAYER │ │  UTILS LAYER     │ │  DB LAYER        │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ Detection        │ │ Validators       │ │ DatabaseManager  │
│ Service          │ │ • License Plate  │ │ • Query builder  │
│ ┌──────────────┐ │ │ • Image          │ │ • Schema mgmt    │
│ │YOLODetector  │ │ │ • Video          │ │ • Indexing       │
│ └──────────────┘ │ │ Helpers          │ │ Models           │
│ ┌──────────────┐ │ │ • Image drawing  │ │ • DetectionRecord│
│ │OCREngine     │ │ │ • ROI extraction │ │ • Statistics     │
│ └──────────────┘ │ │ • IoU calculation│ │ • Sessions       │
│ ┌──────────────┐ │ │ Logging          │ │                  │
│ │ImagePreproc. │ │ │ • Rotating logs  │ │                  │
│ └──────────────┘ │ │ • Levels         │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────────────────┐        ┌─────────────────────────┐
│   EXTERNAL RESOURCES     │        │   DATA & STORAGE        │
├──────────────────────────┤        ├─────────────────────────┤
│ • YOLOv10 Model          │        │ • SQLite Database       │
│ • EasyOCR Model          │        │ • File uploads          │
│ • OpenCV Library         │        │ • Model weights         │
└──────────────────────────┘        │ • Log files             │
                                    └─────────────────────────┘
```

## Component Interaction Flow

```
REQUEST
   │
   ▼
┌─────────────────────────────────┐
│  Flask Route Handler            │
│  (api/routes_v1.py)             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Input Validation               │
│  (utils/validators.py)          │
└────────────┬────────────────────┘
             │ (validated data)
             ▼
┌─────────────────────────────────┐
│  Detection Service              │
│  (services/detection.py)        │
│  - Load image/frame             │
│  - YOLO detection               │
│  - Image preprocessing          │
│  - OCR recognition              │
└────────────┬────────────────────┘
             │ (detection results)
             ▼
┌─────────────────────────────────┐
│  Database Layer                 │
│  (db/models.py)                 │
│  - Create records               │
│  - Store results                │
│  - Update statistics            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  JSON Response                  │
│  (JSON formatted)               │
└────────────┬────────────────────┘
             │
             ▼
           CLIENT
```

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INTERNET / USERS                       │
└────────────────────────┬─────────────────────────────────┘
                         │ (HTTP/HTTPS)
                         │
┌────────────────────────▼─────────────────────────────────┐
│              REVERSE PROXY (Nginx)                        │
│  • Load balancing                                        │
│  • SSL/TLS termination                                   │
│  • Static file serving                                   │
└────────────────────────┬─────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Docker Container 1  │      │  Docker Container 2  │
├──────────────────────┤      ├──────────────────────┤
│ Gunicorn Worker      │      │ Gunicorn Worker      │
│ • Flask App          │      │ • Flask App          │
│ • Port 5001          │      │ • Port 5002          │
└────────┬─────────────┘      └──────────┬───────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────────┐      ┌──────────────────────┐
│   Database Layer     │      │   File Storage       │
│  (SQLite)            │      │   (uploads/)         │
│  • Detections        │      │   • Images           │
│  • Statistics        │      │   • Videos           │
│  • Sessions          │      │                      │
└──────────────────────┘      └──────────────────────┘
```

## Configuration Management

```
┌─────────────────────────────────────────────┐
│     Environment Variables (.env)            │
│  ┌─────────────────────────────────────┐   │
│  │ FLASK_ENV=production                │   │
│  │ SECRET_KEY=***                      │   │
│  │ MODEL_DEVICE=cuda                   │   │
│  │ ... (20+ variables)                 │   │
│  └─────────────────────────────────────┘   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│    src/config.py                            │
│  ┌─────────────────────────────────────┐   │
│  │ Config Class                        │   │
│  │  ├─ DevelopmentConfig               │   │
│  │  ├─ TestingConfig                   │   │
│  │  └─ ProductionConfig                │   │
│  └─────────────────────────────────────┘   │
└────────────────┬────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Flask Application │
        │  Uses config based │
        │  on environment    │
        └────────────────────┘
```

## Data Flow Diagram

```
INPUT SOURCES
  │ Image  │ Video  │ Webcam │
  └─────┬──┴────┬───┴───┬────┘
        │       │       │
        ▼       ▼       ▼
    ┌──────────────────────────┐
    │  Load & Preprocess       │
    │  • Resize                │
    │  • Convert colorspace    │
    │  • Enhance contrast      │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  YOLO Detection          │
    │  Detects: Bounding boxes │
    │          Confidence      │
    │          Class labels    │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  Region of Interest      │
    │  Extract & Prepare ROI   │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  OCR (EasyOCR)           │
    │  Recognizes: License     │
    │              Plate Text  │
    │              Confidence  │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  Validation              │
    │  • Format check          │
    │  • Confidence threshold  │
    │  • Sanity checks         │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  Database Storage        │
    │  • Save to database      │
    │  • Update statistics     │
    │  • Create indexes        │
    └──────────────────────────┘
            │
            ▼
    ┌──────────────────────────┐
    │  API Response            │
    │  JSON with results       │
    └──────────────────────────┘
            │
            ▼
        CLIENT
```

## Module Dependencies

```
run.py (Flask App Factory)
  ├── src/config.py (Configuration)
  ├── src/utils/
  │   ├── logger.py (Logging)
  │   ├── validators.py (Input validation)
  │   └── helpers.py (Utilities)
  ├── src/db/
  │   └── models.py (Database)
  └── src/api/
      ├── errors.py (Error handling)
      ├── routes_v1.py (API endpoints)
      └── services/
          └── detection.py (Business logic)
```

## Testing Architecture

```
┌───────────────────────────────────────┐
│        Test Suite (pytest)            │
├───────────────────────────────────────┤
│ Unit Tests                            │
│  ├─ test_validators.py                │
│  ├─ test_models.py                    │
│  └─ test_services.py                  │
├───────────────────────────────────────┤
│ Integration Tests                     │
│  ├─ test_api.py                       │
│  ├─ test_database.py                  │
│  └─ test_detection_flow.py            │
├───────────────────────────────────────┤
│ Coverage Report                       │
│  • Source coverage > 80%              │
│  • htmlcov/index.html                 │
└───────────────────────────────────────┘
```

## Scalability Considerations

For scaling to production:

```
Single Instance (Current)
    ↓
    ├─ Add load balancer (Nginx/HAProxy)
    ├─ Multiple app instances
    ├─ Redis caching layer
    ├─ PostgreSQL database
    ├─ Celery async tasks
    └─ Monitoring (Prometheus/Grafana)
```

---

**Architecture Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: ✅ Production Ready
