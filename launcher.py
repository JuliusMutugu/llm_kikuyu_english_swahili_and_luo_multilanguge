#!/usr/bin/env python3
"""
Simple launcher for the Trilingual AI system
Starts both API server and Streamlit interface
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_port(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def main():
    print("🚀 Trilingual AI Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ Please run this from the project directory")
        return
    
    # Check API server
    api_running = not check_port(8000)
    if api_running:
        print("✅ API server already running on port 8000")
    else:
        print("🔄 Starting API server...")
        try:
            # Start API server in background
            if os.name == 'nt':  # Windows
                subprocess.Popen([
                    sys.executable, "api_server_modern.py"
                ], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:  # Unix/Linux/Mac
                subprocess.Popen([sys.executable, "api_server_modern.py"])
            
            print("⏳ Waiting for API server to start...")
            time.sleep(3)
            
            # Check if it started
            if not check_port(8000):
                print("✅ API server started successfully")
            else:
                print("⚠️ API server may not have started properly")
        except Exception as e:
            print(f"❌ Failed to start API server: {e}")
            return
    
    # Check Streamlit
    streamlit_running = not check_port(8501)
    if streamlit_running:
        print("✅ Streamlit already running on port 8501")
        print("🌐 Open http://localhost:8501 in your browser")
    else:
        print("🔄 Starting Streamlit interface...")
        try:
            # Start Streamlit
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "streamlit_app.py", 
                "--server.port", "8501",
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false"
            ]
            
            print("🌐 Starting Streamlit at http://localhost:8501")
            print("📱 The interface will open automatically in your browser")
            print("⏹️ Press Ctrl+C to stop")
            
            # Run Streamlit (this will block)
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print("\n⏹️ Streamlit stopped by user")
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")

if __name__ == "__main__":
    main()
