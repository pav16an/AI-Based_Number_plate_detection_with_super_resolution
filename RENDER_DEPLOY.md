# Deploy to Render - Step by Step

## 🚀 Quick Deploy Steps

### 1. Push to GitHub
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/license-plate-detection.git
git push -u origin main
```

### 2. Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Use these settings:
   - **Name**: `license-plate-detection`
   - **Environment**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

### 3. Environment Variables (Optional)
- `FLASK_ENV`: `production`
- `PORT`: `10000` (auto-set by Render)

## 📋 What Gets Deployed
- ✅ Web interface for image upload
- ✅ Live webcam detection
- ✅ Results management with delete
- ✅ REST API endpoints
- ✅ SQLite database

## 🌐 Access Your App
After deployment: `https://your-app-name.onrender.com`

## 🔧 Troubleshooting
- Check build logs in Render dashboard
- Ensure all files are committed to git
- Verify requirements.txt is complete