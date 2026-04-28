@echo off
REM Deployment script for Windows
REM License Plate Detection System

setlocal enabledelayedexpansion

set PROJECT_NAME=license-plate-detection
set ENVIRONMENT=%1
set VERSION=1.0.0

if "%ENVIRONMENT%"=="" set ENVIRONMENT=production

echo.
echo === License Plate Detection System - Deployment Script ===
echo Deployment Configuration:
echo   Project: %PROJECT_NAME%
echo   Environment: %ENVIRONMENT%
echo   Version: %VERSION%
echo.

if "%2"=="test" (
    echo Running tests...
    python -m pytest tests/ -v --cov=src --cov-report=html
) else if "%2"=="docker-build" (
    echo Building Docker image...
    if "%ENVIRONMENT%"=="production" (
        docker build -f docker/Dockerfile -t %PROJECT_NAME%:%VERSION% -t %PROJECT_NAME%:latest .
    ) else (
        docker build -f docker/Dockerfile.dev -t %PROJECT_NAME%:dev-%VERSION% .
    )
) else if "%2"=="docker-run" (
    echo Starting Docker containers...
    if "%ENVIRONMENT%"=="production" (
        docker-compose up -d
    ) else (
        docker-compose -f docker-compose.dev.yml up -d
    )
) else if "%2"=="init-db" (
    echo Initializing database...
    python -c "from src.db import DatabaseManager; from src.config import get_config; config = get_config('%ENVIRONMENT%'); db = DatabaseManager(config.DATABASE_PATH); db.initialize()"
) else if "%2"=="start" (
    echo Starting application...
    if "%ENVIRONMENT%"=="production" (
        gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 "run:create_app()"
    ) else (
        python -m flask run --host 0.0.0.0
    )
) else if "%2"=="full-deploy" (
    echo Running tests...
    python -m pytest tests/ -v --cov=src
    echo Building Docker image...
    docker build -f docker/Dockerfile -t %PROJECT_NAME%:%VERSION% .
    echo Initializing database...
    python -c "from src.db import DatabaseManager; from src.config import get_config; config = get_config('%ENVIRONMENT%'); db = DatabaseManager(config.DATABASE_PATH); db.initialize()"
    echo Starting containers...
    docker-compose up -d
) else (
    echo Usage: %0 [development^|production] [test^|docker-build^|docker-run^|init-db^|start^|full-deploy]
    echo.
    echo Examples:
    echo   %0 production test
    echo   %0 production docker-build
    echo   %0 production full-deploy
)

endlocal
