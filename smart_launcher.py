"""
Smart Launcher for Multilingual Chat System
Handles port conflicts and starts both API and UI
"""

import subprocess
import socket
import time
import sys
import os
from pathlib import Path

def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8000):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + 10):
        if check_port_available(port):
            return port
    return None

def start_api_server():
    """Start the API server"""
    print("🚀 Starting API server...")
    
    # Check if already running
    for port in range(8000, 8010):
        if not check_port_available(port):
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/", timeout=1)
                if response.status_code == 200:
                    print(f"✅ API server already running on port {port}")
                    return port
            except:
                continue
    
    # Start new API server
    try:
        api_process = subprocess.Popen([
            sys.executable, "multi_model_api.py"
        ], cwd=Path(__file__).parent)
        
        # Wait a moment for it to start
        time.sleep(3)
        
        # Find which port it's using
        for port in range(8000, 8010):
            if not check_port_available(port):
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/", timeout=1)
                    if response.status_code == 200:
                        print(f"✅ API server started on port {port}")
                        return port
                except:
                    continue
        
        print("❌ Failed to start API server")
        return None
        
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def start_streamlit_app():
    """Start the Streamlit app"""
    print("🎨 Starting Streamlit interface...")
    
    try:
        # Find available port for Streamlit
        streamlit_port = find_available_port(8501)
        if not streamlit_port:
            print("❌ No available ports for Streamlit")
            return
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "minimal_chat_app.py",
            "--server.port", str(streamlit_port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], cwd=Path(__file__).parent)
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

def main():
    print("=" * 60)
    print("🌍 MULTILINGUAL CHAT SYSTEM LAUNCHER")
    print("=" * 60)
    
    # Check Python environment
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working Directory: {Path(__file__).parent}")
    
    # Start API server first
    api_port = start_api_server()
    if not api_port:
        print("\n❌ Cannot start system without API server")
        print("💡 Try running: python port_manager.py")
        return
    
    print(f"\n📚 API Documentation: http://localhost:{api_port}/docs")
    
    # Start Streamlit interface
    print("\n" + "="*60)
    start_streamlit_app()

if __name__ == "__main__":
    main()
