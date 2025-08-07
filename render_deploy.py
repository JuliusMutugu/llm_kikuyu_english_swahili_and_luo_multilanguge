#!/usr/bin/env python3
"""
Render Deployment Helper
Guides you through deploying on Render.com
"""

import os
import webbrowser
import time
from pathlib import Path

def render_deployment_guide():
    """Interactive guide for Render deployment"""
    
    print("🚀 Render Deployment Helper - Trilingual AI Assistant")
    print("=" * 60)
    
    # Check files
    required_files = [
        'multi_model_api.py',
        'streamlit_app.py', 
        'api_requirements_render.txt',
        'requirements.txt',
        'render.yaml'
    ]
    
    print("\n📋 Checking required files...")
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️ Missing files: {missing}")
        print("Please ensure all files are present before deploying.")
        return
    
    print("\n✅ All files ready for deployment!")
    
    # GitHub check
    print("\n📁 GitHub Repository Setup:")
    print("1. Make sure your code is pushed to GitHub")
    print("2. Repository should be public (for free tier)")
    print("3. All files should be in the root directory")
    
    input("\nPress Enter when your GitHub repo is ready...")
    
    # Render deployment steps
    print("\n🌟 Render Deployment Steps:")
    print("=" * 40)
    
    print("\nStep 1: Open Render Dashboard")
    print("Going to render.com...")
    webbrowser.open("https://render.com")
    
    input("\nPress Enter after you've signed up/logged in with GitHub...")
    
    print("\nStep 2: Create API Service")
    print("🔧 API Service Configuration:")
    print("- Name: trilingual-ai-api")
    print("- Repository: Select your GitHub repo")
    print("- Branch: main")
    print("- Build Command: pip install -r api_requirements_render.txt")
    print("- Start Command: uvicorn multi_model_api:app --host 0.0.0.0 --port $PORT")
    print("- Plan: Free")
    
    input("\nPress Enter after creating the API service...")
    
    print("\nStep 3: Create UI Service")
    print("🎨 UI Service Configuration:")
    print("- Name: trilingual-ai-ui")
    print("- Repository: Same GitHub repo")
    print("- Branch: main")
    print("- Build Command: pip install -r requirements.txt")
    print("- Start Command: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true")
    print("- Plan: Free")
    print("- Environment Variable: API_URL = https://trilingual-ai-api.onrender.com")
    
    input("\nPress Enter after creating the UI service...")
    
    print("\nStep 4: Monitor Deployment")
    print("📊 Check the deployment logs in Render dashboard")
    print("⏱️ Build time: Usually 3-8 minutes")
    print("🎯 Once deployed, you'll get URLs like:")
    print("   API: https://trilingual-ai-api.onrender.com")
    print("   UI:  https://trilingual-ai-ui.onrender.com")
    
    print("\nStep 5: Test Your Deployment")
    print("🧪 Test these URLs:")
    print("   Health: https://trilingual-ai-api.onrender.com/health")
    print("   Docs:   https://trilingual-ai-api.onrender.com/docs")
    print("   Chat:   https://trilingual-ai-ui.onrender.com")
    
    print("\n🎉 Deployment Complete!")
    print("Your Trilingual AI Assistant is now live on Render!")
    print("\n💡 Pro Tips:")
    print("- Services sleep after 15 min of inactivity (free tier)")
    print("- First request after sleep takes 30-60 seconds")
    print("- Push code updates to auto-redeploy")
    print("- Monitor usage in Render dashboard")

def main():
    """Main function"""
    try:
        render_deployment_guide()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment helper cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
