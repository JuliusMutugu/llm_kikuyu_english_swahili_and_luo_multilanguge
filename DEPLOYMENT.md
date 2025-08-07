# Simple deployment guide

## 🚀 Quick Start (Local Development)

### Option 1: Automatic Launcher (Recommended)
```bash
python launch_system.py
```
This will start both the API server and Streamlit interface automatically.

### Option 2: Manual Start
```bash
# Terminal 1 - Start API Server
python -m uvicorn multi_model_api:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 - Start Streamlit Interface  
streamlit run streamlit_app.py
```

### Option 3: Using Scripts
```bash
# On Windows
start_api.bat

# On Linux/Mac
chmod +x start_api.sh
./start_api.sh
```

## 🐳 Docker Deployment

### Build and run with Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Build and run API only
```bash
docker build -t trilingual-api .
docker run -p 8001:8001 trilingual-api
```

## ☁️ Cloud Deployment

### Railway (Recommended for quick deployment)
1. Connect your GitHub repository to Railway
2. Deploy automatically - Railway will detect the configuration
3. Your API will be available at: `https://your-app.railway.app`

### Heroku
```bash
# Install Heroku CLI, then:
heroku create your-app-name
git push heroku main
```

### Render
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use: `uvicorn multi_model_api:app --host 0.0.0.0 --port $PORT`

### Google Cloud Run
```bash
gcloud run deploy trilingual-api --source . --platform managed --region us-central1 --allow-unauthenticated
```

## 📡 API Endpoints

Once deployed, your API will have these endpoints:
- `GET /` - Health check and info
- `GET /health` - Health status
- `POST /chat` - Main chat endpoint
- `GET /status` - Detailed system status
- `GET /docs` - Interactive API documentation

## 🎯 Testing Deployment

Test with curl:
```bash
curl -X POST "http://localhost:8001/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello, how are you?", "language": "auto"}'
```

## 🔧 Environment Variables (for cloud deployment)

- `PORT` - Port number (default: 8001)
- `PYTHONPATH` - Python path (set to `/app`)
- `MODEL_PATH` - Path to model files (optional)

## 🎨 Streamlit Interface

After API is running, access the chat interface at:
- Local: http://localhost:8501
- Cloud: Deploy Streamlit separately or use the Docker Compose setup

## 🛠️ Troubleshooting

### Common Issues:
1. **Port already in use**: Change port in the startup commands
2. **Missing dependencies**: Run `pip install -r api_requirements.txt`
3. **Model not found**: The API will use fallback responses if models are not available
4. **Memory issues**: Consider using smaller model variants for deployment

### Logs:
- API logs will show in the terminal/container logs
- Check `/health` endpoint for system status
- Use `/docs` for interactive API testing
