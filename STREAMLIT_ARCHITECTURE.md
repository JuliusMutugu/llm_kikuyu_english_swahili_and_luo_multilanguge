# 🏗️ Streamlit App Architecture with Deployed URL

## 🎯 Current Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  🌐 https://llm-kikuyu-english-swahili-and-luo.onrender.com │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Streamlit UI  │◄──►│       FastAPI Backend          │ │
│  │                 │    │                                 │ │
│  │ • Chat Interface│    │ • /chat endpoint               │ │
│  │ • Language Menu │    │ • /health endpoint             │ │
│  │ • Settings      │    │ • /docs endpoint               │ │
│  │ • File uploads  │    │ • Language detection           │ │
│  └─────────────────┘    │ • Response generation          │ │
│                         └─────────────────────────────────┘ │
│                                                             │
│              🔄 Combined Service Deployment                 │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 How API URL Detection Works

### 1. **Priority Order (Smart Detection):**
```python
def get_api_url():
    # 🥇 Priority 1: Explicit API_URL environment variable
    if os.environ.get('API_URL'):
        return API_URL
    
    # 🥈 Priority 2: Render platform detection
    if os.environ.get('RENDER_EXTERNAL_URL'):
        return RENDER_EXTERNAL_URL
    
    # 🥉 Priority 3: Auto-detect cloud platform
    if 'onrender.com' in hostname:
        return 'https://llm-kikuyu-english-swahili-and-luo.onrender.com'
    
    # 🔧 Priority 4: Local development
    return 'http://localhost:8001'
```

### 2. **Your Current Setup:**
- **Deployment Type**: Combined Service (UI + API in one)
- **Platform**: Render.com
- **URL**: https://llm-kikuyu-english-swahili-and-luo.onrender.com
- **API Base**: Same URL as UI (combined)

## 🚀 How It Works in Practice

### **When User Opens Your App:**

1. **Browser loads**: `https://llm-kikuyu-english-swahili-and-luo.onrender.com`
2. **Streamlit detects**: Running on Render platform
3. **API URL set to**: `https://llm-kikuyu-english-swahili-and-luo.onrender.com`
4. **Chat requests go to**: `https://llm-kikuyu-english-swahili-and-luo.onrender.com/chat`

### **Request Flow:**
```
User types message
       ↓
Streamlit UI captures input
       ↓
POST to /chat endpoint on same domain
       ↓
FastAPI processes request
       ↓
Response sent back to UI
       ↓
UI displays response
```

## ✅ Benefits of Your Current Setup

### **🎯 Advantages:**
- ✅ **Simple deployment** - One service handles everything
- ✅ **No CORS issues** - Same domain for UI and API
- ✅ **Cost effective** - Single free tier service
- ✅ **Easy maintenance** - One service to monitor
- ✅ **Fast communication** - No network latency between services

### **🔧 Technical Details:**
- **Start Command**: Likely runs both Streamlit and FastAPI
- **Port Sharing**: Both services on same $PORT
- **Route Handling**: FastAPI serves API, Streamlit serves UI
- **Domain**: Single domain handles all traffic

## 📊 Available Endpoints

Your deployed app provides these endpoints:

```
🌐 https://llm-kikuyu-english-swahili-and-luo.onrender.com/
├── / ........................... Streamlit UI (Main interface)
├── /chat ....................... POST - Chat API endpoint
├── /health ..................... GET - Health check
├── /docs ....................... GET - API documentation
└── /_stcore/* .................. Streamlit internal endpoints
```

## 🧪 Testing Your Current Setup

To verify everything works correctly:

```python
# Test API endpoint
import requests

base_url = "https://llm-kikuyu-english-swahili-and-luo.onrender.com"

# Test health
response = requests.get(f"{base_url}/health")
print(f"Health: {response.status_code}")

# Test chat
chat_data = {
    "message": "Habari yako?",
    "language": "kiswahili"
}
response = requests.post(f"{base_url}/chat", json=chat_data)
print(f"Chat: {response.status_code}")
```

## 🔄 How to Update/Modify

If you need to make changes:

1. **Update code locally**
2. **Commit to GitHub**
3. **Render auto-deploys** from your repository
4. **Changes go live** automatically

## 🎉 Your Setup is Optimized!

Your current architecture is actually **perfect** for a free deployment:
- Simple and efficient
- Cost-effective
- Easy to maintain
- Production-ready

The Streamlit app automatically detects it's running on Render and uses the correct URL for API calls!
