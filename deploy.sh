#!/bin/bash

# License Plate Detection System Deployment Script

echo "🚀 Starting deployment of License Plate Detection System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads json weights

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start containers
echo "🔨 Building and starting containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Deployment successful!"
    echo "🌐 Application is running at: http://localhost"
    echo "📊 Direct Flask access: http://localhost:5000"
    echo ""
    echo "📋 Available endpoints:"
    echo "   - Home: http://localhost/"
    echo "   - Live Detection: http://localhost/webcam"
    echo "   - Results: http://localhost/results"
    echo "   - API: http://localhost/api/plates"
    echo ""
    echo "🔧 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Deployment failed. Check logs with: docker-compose logs"
    exit 1
fi