"""
Complete System Launcher
Launches the conversational system with multi-chat functionality
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server"""
    print("🚀 Starting API Server...")
    
    # Try the multi-model API first
    if os.path.exists("multi_model_api.py"):
        cmd = [sys.executable, "multi_model_api.py"]
    elif os.path.exists("api_server_modern.py"):
        cmd = [sys.executable, "api_server_modern.py"]
    else:
        print("❌ No API server found!")
        return None
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ API server started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def start_streamlit_app():
    """Start the Streamlit application"""
    print("🌐 Starting Streamlit App...")
    
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
        process = subprocess.Popen(cmd)
        print(f"✅ Streamlit app started with PID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")
        return None

def main():
    """Main launcher function"""
    print("🚀 Launching Complete Conversational System...")
    print("=" * 50)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Cannot continue without API server")
        return
    
    # Wait a moment for API to start
    print("⏳ Waiting for API server to start...")
    time.sleep(3)
    
    # Check if API is responding
    for i in range(10):
        if check_api_server():
            print("✅ API server is online!")
            break
        time.sleep(1)
        print(f"⏳ Waiting for API... ({i+1}/10)")
    else:
        print("❌ API server failed to start properly")
        api_process.terminate()
        return
    
    # Start Streamlit app
    streamlit_process = start_streamlit_app()
    if not streamlit_process:
        print("❌ Failed to start Streamlit app")
        api_process.terminate()
        return
    
    print("\n🎉 System launched successfully!")
    print("=" * 50)
    print("🌐 Streamlit App: http://localhost:8501")
    print("🔧 API Server: http://localhost:8000") 
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n💡 Features available:")
    print("   • Multi-chat conversations")
    print("   • Multilingual support (English, Kiswahili, Kikuyu, Luo)")
    print("   • Modern UI with chat management")
    print("   • Export/import functionality")
    print("   • Real-time API communication")
    print("\n⚡ To stop: Press Ctrl+C")
    print("=" * 50)
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
                
            if streamlit_process.poll() is not None:
                print("❌ Streamlit app stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        
        # Terminate processes
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            print("✅ Streamlit app stopped")
            
        if api_process and api_process.poll() is None:
            api_process.terminate()
            print("✅ API server stopped")
        
        print("👋 System shutdown complete!")

if __name__ == "__main__":
    main()
