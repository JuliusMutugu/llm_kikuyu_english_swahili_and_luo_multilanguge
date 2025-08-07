#!/bin/bash
# Local API startup script

echo "🚀 Starting Trilingual AI API Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r api_requirements.txt

# Start the API server
echo "🌐 Starting API server on port 8001..."
python -m uvicorn multi_model_api:app --host 0.0.0.0 --port 8001 --reload

echo "✅ API Server started successfully!"
echo "📖 API Documentation: http://localhost:8001/docs"
echo "💬 Chat Interface: Run 'streamlit run streamlit_app.py' in another terminal"
