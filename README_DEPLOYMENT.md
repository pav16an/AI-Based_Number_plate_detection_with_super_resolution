# License Plate Detection System - Deployment Guide

## 🚀 Quick Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Prerequisites
- Docker Desktop installed
- Docker Compose installed
- At least 4GB RAM available
- 2GB free disk space

#### Windows Deployment
```bash
# Run the deployment script
deploy.bat
```

#### Linux/Mac Deployment
```bash
# Make script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

#### Manual Docker Deployment
```bash
# Create directories
mkdir -p uploads json weights

# Build and run with Docker Compose
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Local Python Deployment

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation Steps
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python debug_db.py

# Run the application
python app.py
```

#### Production with Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -c gunicorn.conf.py app:app
```

### Option 3: Cloud Deployment

#### Heroku Deployment
1. Create a Heroku app
2. Add the following buildpacks:
   - `heroku/python`
   - `https://github.com/heroku/heroku-buildpack-apt`
3. Create `Aptfile` with system dependencies
4. Deploy using Git

#### AWS EC2 Deployment
1. Launch an EC2 instance (t3.medium or larger recommended)
2. Install Docker and Docker Compose
3. Clone the repository
4. Run the deployment script

#### Google Cloud Run Deployment
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/license-plate-detection

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/license-plate-detection --platform managed
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `FLASK_DEBUG`: Set to `False` for production
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)

### Model Configuration
- Place your custom YOLO model in the `weights/` directory as `best.pt`
- The system will fallback to YOLOv8n if custom model is not available

### Database Configuration
- SQLite database is used by default
- Database file: `licensePlatesDatabase.db`
- Automatic table creation and migration

## 📊 Monitoring and Maintenance

### Health Checks
- Application health: `http://localhost:5000/`
- Database check: `http://localhost:5000/check_db`
- API status: `http://localhost:5000/api/plates`

### Logs
```bash
# Docker logs
docker-compose logs -f

# Application logs
tail -f /var/log/license-plate-detection.log
```

### Backup
```bash
# Backup database
cp licensePlatesDatabase.db backup_$(date +%Y%m%d_%H%M%S).db

# Backup uploads
tar -czf uploads_backup_$(date +%Y%m%d_%H%M%S).tar.gz uploads/
```

## 🌐 Access Points

After successful deployment, the application will be available at:

- **Main Application**: http://localhost (with nginx) or http://localhost:5000 (direct)
- **Home Page**: Upload images for detection
- **Live Detection**: Real-time webcam detection at `/webcam`
- **Results Page**: View all detected license plates at `/results`
- **API Endpoint**: RESTful API at `/api/plates`

## 🔒 Security Considerations

### Production Security
- Change default ports
- Enable HTTPS with SSL certificates
- Implement authentication for admin features
- Set up firewall rules
- Regular security updates

### File Upload Security
- File type validation implemented
- File size limits enforced
- Secure file storage

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :5000
   
   # Kill the process or change port in docker-compose.yml
   ```

2. **Model Loading Issues**
   ```bash
   # Check if weights file exists
   ls -la weights/
   
   # Download YOLOv8 model if needed
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

3. **Database Issues**
   ```bash
   # Reset database
   rm licensePlatesDatabase.db
   python debug_db.py
   ```

4. **Memory Issues**
   ```bash
   # Reduce worker count in gunicorn.conf.py
   # Increase Docker memory allocation
   ```

### Performance Optimization

1. **CPU Optimization**
   - Adjust worker count in gunicorn configuration
   - Use CPU-optimized YOLO models

2. **Memory Optimization**
   - Implement image resizing before processing
   - Clear processed images from memory

3. **Storage Optimization**
   - Implement automatic cleanup of old uploads
   - Compress stored images

## 📞 Support

For deployment issues or questions:
1. Check the logs first
2. Verify all prerequisites are met
3. Ensure sufficient system resources
4. Check firewall and network settings

## 🔄 Updates

To update the application:
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```