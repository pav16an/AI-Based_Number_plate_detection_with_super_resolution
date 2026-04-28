# Deployment Checklist

## Pre-Deployment Checklist

### 1. Code Quality
- [ ] All tests passing: `pytest tests/ -v`
- [ ] No linting errors: `flake8 src`
- [ ] Type checking passes: `mypy src`
- [ ] Coverage above 80%: `pytest tests/ --cov=src`
- [ ] Code reviewed by team member

### 2. Configuration
- [ ] `.env` file created from `.env.example`
- [ ] All required environment variables set
- [ ] `SECRET_KEY` is strong (production)
- [ ] `DEBUG = False` (production)
- [ ] Database path correct
- [ ] Model weights path correct

### 3. Dependencies
- [ ] `requirements-prod.txt` installed and tested
- [ ] No deprecated packages used
- [ ] All dependencies have compatible versions
- [ ] GPU libraries installed (if using GPU)

### 4. Database
- [ ] Database initialized: `python init_project.py`
- [ ] Migrations applied
- [ ] Backup created (if migrating)
- [ ] Database user/password changed (if applicable)

### 5. Model & Weights
- [ ] Model file exists: `weights/best.pt`
- [ ] Model file not corrupted
- [ ] Model can be loaded without errors
- [ ] Weights are correct version

### 6. Security
- [ ] No secrets in code
- [ ] Secrets in environment variables only
- [ ] CORS origins configured
- [ ] Rate limiting enabled
- [ ] File upload validation enabled
- [ ] SQL injection prevention verified

### 7. Logging
- [ ] Log directory exists: `logs/`
- [ ] Log level appropriate for environment
- [ ] Log rotation configured
- [ ] Log files have proper permissions

### 8. Documentation
- [ ] README updated with deployment info
- [ ] API documentation current
- [ ] Environment variables documented
- [ ] Troubleshooting guide complete

### 9. Docker (if using)
- [ ] Dockerfile builds successfully
- [ ] `docker-compose.yml` configured
- [ ] Health checks defined
- [ ] Port mappings correct
- [ ] Volume mounts verified

### 10. Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] API endpoints tested with curl
- [ ] Error scenarios tested
- [ ] Load testing completed (if applicable)

---

## Deployment Steps

### Step 1: Final Validation
```bash
# Run all checks
pytest tests/ -v --cov=src
flake8 src
python -c "from src.db import DatabaseManager; print('Import OK')"
python -c "from src.services import DetectionService; print('Import OK')"
```

### Step 2: Database Backup (if applicable)
```bash
cp data/app.db data/app.db.backup.$(date +%Y%m%d)
```

### Step 3: Initialize/Migrate Database
```bash
python init_project.py
```

### Step 4: Start Application

**Option A: Local Python**
```bash
export FLASK_ENV=production
python run.py
```

**Option B: Gunicorn**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 'run:create_app()'
```

**Option C: Docker**
```bash
docker build -f docker/Dockerfile -t license-plate:1.0.0 .
docker-compose up -d
```

### Step 5: Verify Deployment
```bash
# Health check
curl http://localhost:5000/health

# API health
curl http://localhost:5000/api/v1/health

# Check logs
docker-compose logs -f app
# or
tail -f logs/app.log
```

### Step 6: Smoke Tests
```bash
# Test basic API functionality
curl http://localhost:5000/api/v1/detections?page=1

# Test with a test image
curl -X POST -F "file=@test_image.jpg" \
  http://localhost:5000/api/v1/detect/image
```

---

## Post-Deployment Verification

- [ ] Application is running
- [ ] API endpoints respond correctly
- [ ] Database is working
- [ ] Log files are being written
- [ ] File uploads work
- [ ] Detection works
- [ ] No error messages in logs
- [ ] Response times are acceptable

---

## Rollback Plan

### Quick Rollback
```bash
# Stop current deployment
docker-compose down

# Restore from backup
docker build -f docker/Dockerfile -t license-plate:previous .
docker-compose up -d
```

### Database Rollback
```bash
# Restore database backup
cp data/app.db.backup.20240101 data/app.db

# Restart application
docker-compose restart app
```

---

## Monitoring Post-Deployment

### Daily Checks
```bash
# Check application health
curl http://localhost:5000/health

# Review error logs
grep ERROR logs/app.log | tail -20

# Check database size
du -h data/app.db

# Monitor system resources
docker stats license-plate-detection
```

### Weekly Checks
- [ ] Performance metrics reviewed
- [ ] Error rates checked
- [ ] Database optimization run
- [ ] Security logs reviewed
- [ ] Backup integrity verified

---

## Troubleshooting During Deployment

### Application Won't Start
```bash
# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip list | grep flask

# Check configuration
cat .env

# Check logs for errors
tail -50 logs/app.log
```

### Database Errors
```bash
# Reinitialize database
rm data/app.db
python init_project.py

# Check database file
ls -lh data/app.db
```

### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or change port in .env
PORT=5001
```

---

## Success Criteria

✅ Application running without errors  
✅ API responding to requests  
✅ Database storing data correctly  
✅ Logs being written  
✅ Performance acceptable  
✅ No security warnings  
✅ Team notified of successful deployment  

---

## Documentation Update

- [ ] Update deployment date in docs
- [ ] Document any issues encountered
- [ ] Update version number if applicable
- [ ] Add entries to CHANGELOG
- [ ] Notify team via Slack/Email

---

**Deployment Date**: ___________  
**Deployed By**: ___________  
**Version**: 1.0.0  
**Status**: ⏳ Pending / ✅ Complete / ❌ Rollback
