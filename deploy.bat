@echo off
REM License Plate Detection System Deployment Script for Windows

echo 🚀 Starting deployment of License Plate Detection System...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "json" mkdir json
if not exist "weights" mkdir weights

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start containers
echo 🔨 Building and starting containers...
docker-compose up --build -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
docker-compose ps | findstr "Up" >nul
if %errorlevel% equ 0 (
    echo ✅ Deployment successful!
    echo 🌐 Application is running at: http://localhost
    echo 📊 Direct Flask access: http://localhost:5000
    echo.
    echo 📋 Available endpoints:
    echo    - Home: http://localhost/
    echo    - Live Detection: http://localhost/webcam
    echo    - Results: http://localhost/results
    echo    - API: http://localhost/api/plates
    echo.
    echo 🔧 To view logs: docker-compose logs -f
    echo 🛑 To stop: docker-compose down
) else (
    echo ❌ Deployment failed. Check logs with: docker-compose logs
    pause
    exit /b 1
)

pause