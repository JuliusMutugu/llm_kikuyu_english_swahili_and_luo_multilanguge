#!/bin/bash
# Docker Deployment Script with Enhanced Learning

echo "=========================================="
echo "  Trilingual AI - Enhanced Docker Deploy"
echo "=========================================="
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to build Docker images!"
    exit 1
fi

echo "✅ Docker images built successfully"
echo

# Start core services
echo "🚀 Starting core services..."
docker-compose up -d trilingual-api streamlit-ui

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to start services!"
    exit 1
fi

echo "✅ Core services started"
echo

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check API health
API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health 2>/dev/null)
if [ "$API_HEALTH" = "200" ]; then
    echo "✅ API Server: Healthy"
else
    echo "⚠️  API Server: Starting... (may take a few more seconds)"
fi

# Check Streamlit health
STREAMLIT_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501/_stcore/health 2>/dev/null)
if [ "$STREAMLIT_HEALTH" = "200" ]; then
    echo "✅ Streamlit UI: Healthy"
else
    echo "⚠️  Streamlit UI: Starting... (may take a few more seconds)"
fi

echo

# Initialize dictionary learning (optional)
echo "📚 Initialize Luo dictionary learning? (y/n)"
read -r INIT_LEARNING

if [ "$INIT_LEARNING" = "y" ] || [ "$INIT_LEARNING" = "Y" ]; then
    echo "🔄 Running initial dictionary learning..."
    docker-compose --profile learning run --rm dictionary-learner
    
    if [ $? -eq 0 ]; then
        echo "✅ Dictionary learning completed"
    else
        echo "⚠️  Dictionary learning had issues (check logs)"
    fi
fi

echo
echo "=========================================="
echo "🎉 Deployment Complete!"
echo "=========================================="
echo
echo "📱 Access your Enhanced Trilingual AI:"
echo "   🌐 Web Interface: http://localhost:8501"
echo "   🔌 API Server:    http://localhost:8001"
echo
echo "🔧 Enhanced Features Available:"
echo "   ✅ Federated Learning from online sources"
echo "   ✅ Luo dictionary learning from Glosbe.com"
echo "   ✅ Advanced analytics and feedback"
echo "   ✅ Privacy-preserving vocabulary expansion"
echo
echo "📊 Useful Commands:"
echo "   docker-compose logs -f streamlit-ui  # View UI logs"
echo "   docker-compose logs -f trilingual-api  # View API logs"
echo "   docker-compose down  # Stop all services"
echo "   docker-compose --profile learning run dictionary-learner  # Run learning manually"
echo
echo "🚀 Your enhanced AI is ready!"
echo "   Open http://localhost:8501 and check the 'Learning' tab!"
echo
