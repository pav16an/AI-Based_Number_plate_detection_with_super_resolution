#!/bin/bash
# Deployment script for License Plate Detection System

set -e

echo "=== License Plate Detection System - Deployment Script ==="

# Configuration
PROJECT_NAME="license-plate-detection"
ENVIRONMENT=${1:-"production"}
VERSION="1.0.0"

echo "Deployment Configuration:"
echo "  Project: $PROJECT_NAME"
echo "  Environment: $ENVIRONMENT"
echo "  Version: $VERSION"

# Function to run tests
run_tests() {
    echo ""
    echo "Running tests..."
    python -m pytest tests/ -v --cov=src --cov-report=html
}

# Function to build Docker image
build_docker() {
    echo ""
    echo "Building Docker image..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker build -f docker/Dockerfile -t $PROJECT_NAME:$VERSION -t $PROJECT_NAME:latest .
    else
        docker build -f docker/Dockerfile.dev -t $PROJECT_NAME:dev-$VERSION .
    fi
}

# Function to run Docker containers
run_docker() {
    echo ""
    echo "Starting Docker containers..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker-compose up -d
    else
        docker-compose -f docker-compose.dev.yml up -d
    fi
}

# Function to initialize database
init_db() {
    echo ""
    echo "Initializing database..."
    python -c "from src.db import DatabaseManager; from src.config import get_config; config = get_config('$ENVIRONMENT'); db = DatabaseManager(config.DATABASE_PATH); db.initialize()"
}

# Function to start application
start_app() {
    echo ""
    echo "Starting application..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 'run:create_app()'
    else
        python -m flask run --host 0.0.0.0
    fi
}

# Main execution
case "$2" in
    "test")
        run_tests
        ;;
    "docker-build")
        build_docker
        ;;
    "docker-run")
        run_docker
        ;;
    "init-db")
        init_db
        ;;
    "start")
        start_app
        ;;
    "full-deploy")
        run_tests
        build_docker
        init_db
        run_docker
        ;;
    *)
        echo "Usage: $0 [development|production] [test|docker-build|docker-run|init-db|start|full-deploy]"
        echo ""
        echo "Examples:"
        echo "  $0 production test"
        echo "  $0 production docker-build"
        echo "  $0 production docker-run"
        echo "  $0 production full-deploy"
        exit 1
        ;;
esac

echo ""
echo "=== Deployment Complete ==="
