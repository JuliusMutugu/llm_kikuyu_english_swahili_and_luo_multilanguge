"""
Simple Launcher for Minimal Chat System
"""

import subprocess
import sys
import time
import requests
import os

def check_api():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api():
    """Start the API server"""
    print("🚀 Starting API Server...")
    try:
        process = subprocess.Popen([sys.executable, "multi_model_api.py"])
        print(f"✅ API started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return None

def start_chat_app():
    """Start the minimal chat app"""
    print("💬 Starting Chat Interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "minimal_chat_app.py", "--server.port", "8501"])
    except Exception as e:
        print(f"❌ Failed to start chat app: {e}")

def main():
    """Main launcher"""
    print("🚀 Launching Minimal Multilingual Chat System")
    print("=" * 50)
    
    # Start API
    api_process = start_api()
    if not api_process:
        return
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    for i in range(10):
        if check_api():
            print("✅ API is ready!")
            break
        time.sleep(1)
    else:
        print("❌ API failed to start")
        api_process.terminate()
        return
    
    print("\n🌐 Opening chat interface...")
    print("   Chat App: http://localhost:8501")
    print("   API: http://localhost:8000")
    print("\n💬 Features:")
    print("   • Simple, clean interface")
    print("   • Responds in selected language")
    print("   • 4 languages supported")
    print("   • Real-time conversation")
    
    # Start chat app (this will block)
    start_chat_app()

if __name__ == "__main__":
    main()
