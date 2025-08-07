# 🛠️ Render Troubleshooting Guide

## 🔧 Common Render Issues & Solutions

### 1. Build Failures

#### Problem: "Build failed" or timeout
**Solutions:**
- Check build logs for specific error messages
- Ensure `api_requirements_render.txt` has lightweight packages only
- Verify Python version compatibility (3.8-3.11 supported)
- Remove heavy dependencies like `torch`, `tensorflow`

#### Problem: "pip install failed"
**Solutions:**
```bash
# Use specific versions that work well on Render
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.2
```

### 2. Service Start Failures

#### Problem: Service won't start
**Check these:**
- Start command syntax: `uvicorn multi_model_api:app --host 0.0.0.0 --port $PORT`
- File exists: `multi_model_api.py` in root directory
- App variable: Make sure your FastAPI app is named `app`

#### Problem: Port binding errors
**Solution:**
Always use `$PORT` environment variable:
```python
# In your app
port = int(os.environ.get("PORT", 8000))
```

### 3. Service Communication Issues

#### Problem: UI can't connect to API
**Solutions:**
- Check API_URL environment variable in UI service
- Use full HTTPS URL: `https://trilingual-ai-api.onrender.com`
- Wait for API service to be fully deployed before deploying UI

#### Problem: CORS errors
**Add to your FastAPI app:**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 4. Memory Issues

#### Problem: Service crashes with memory errors
**Solutions:**
- Remove unused imports
- Use lightweight libraries only
- Implement basic fallback responses instead of heavy ML models

### 5. Cold Starts

#### Problem: Service takes long to respond after inactivity
**This is normal on free tier:**
- Services sleep after 15 minutes of inactivity
- First request takes 30-60 seconds to wake up
- Consider upgrading to paid tier for always-on services

### 6. Environment Variables

#### Problem: Environment variables not working
**Check:**
- Variable name spelling (case sensitive)
- No quotes around values in Render dashboard
- Restart service after adding variables

## 🚀 Performance Optimization

### 1. Faster Builds
```yaml
# In render.yaml
buildCommand: |
  pip install --no-cache-dir -r api_requirements_render.txt
```

### 2. Health Checks
```python
# Add to your FastAPI app
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

### 3. Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 📊 Monitoring Your Deployment

### Render Dashboard Features:
- **Metrics**: CPU, memory, response times
- **Logs**: Real-time application logs  
- **Events**: Deployment history
- **Settings**: Environment variables, scaling

### Key Metrics to Watch:
- **Build time**: Should be under 10 minutes
- **Memory usage**: Stay under 512MB (free tier)
- **Response time**: API should respond in <1 second
- **Error rate**: Keep below 1%

## 🆘 When Things Go Wrong

### 1. Check Logs First
- Go to service in Render dashboard
- Click "Logs" tab
- Look for error messages and stack traces

### 2. Common Error Messages

#### "ModuleNotFoundError"
- Missing dependency in requirements file
- Typo in import statement

#### "Address already in use"
- Wrong port configuration
- Use `$PORT` environment variable

#### "Application startup failed"
- Check FastAPI app initialization
- Verify all imports work

### 3. Debug Locally First
```bash
# Test your app locally before deploying
pip install -r api_requirements_render.txt
uvicorn multi_model_api:app --host 0.0.0.0 --port 8000
```

## 📞 Getting Help

### Render Support:
- Documentation: https://render.com/docs
- Community: https://community.render.com
- Status page: https://status.render.com

### Quick Fixes:
1. **Restart service** - Often fixes temporary issues
2. **Clear build cache** - In service settings
3. **Redeploy** - Push a small change to trigger redeploy
4. **Check status page** - Verify Render isn't having issues

## ✅ Success Checklist

Before declaring success, verify:
- [ ] API service shows "Live" status
- [ ] UI service shows "Live" status  
- [ ] Health endpoint responds: `/health`
- [ ] API docs load: `/docs`
- [ ] Streamlit UI loads completely
- [ ] Chat functionality works end-to-end
- [ ] Both services communicate properly

Your Trilingual AI Assistant should now be running smoothly on Render! 🎉
