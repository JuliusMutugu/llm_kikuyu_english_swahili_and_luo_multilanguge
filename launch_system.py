#!/usr/bin/env python3
"""
Trilingual AI System Launcher
Starts both API server and Streamlit interface
"""

import subprocess
import sys
import time
import requests
import os
import signal
import threading
from pathlib import Path

class TrilingualLauncher:
    def __init__(self):
        self.api_process = None
        self.streamlit_process = None
        self.running = True
        
    def check_port(self, port):
        """Check if a port is available"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def wait_for_api(self, max_attempts=30):
        """Wait for API to be ready"""
        print("🔄 Waiting for API to start...")
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:8001/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API is ready!")
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def start_api(self):
        """Start the API server"""
        print("🚀 Starting API server...")
        
        # Check if port is available
        if not self.check_port(8001):
            print("⚠️ Port 8001 is already in use. API might already be running.")
            return True
        
        try:
            # Install requirements if needed
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "api_requirements.txt"], 
                         check=True, capture_output=True)
            
            # Start API
            self.api_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "multi_model_api:app", 
                "--host", "0.0.0.0", 
                "--port", "8001",
                "--reload"
            ])
            
            return self.wait_for_api()
            
        except Exception as e:
            print(f"❌ Failed to start API: {e}")
            return False
    
    def start_streamlit(self):
        """Start the Streamlit interface"""
        print("🎨 Starting Streamlit interface...")
        
        # Check if port is available
        if not self.check_port(8501):
            print("⚠️ Port 8501 is already in use. Streamlit might already be running.")
            return True
        
        try:
            # Install requirements if needed
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
            
            # Start Streamlit
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "streamlit_app.py",
                "--server.address", "0.0.0.0",
                "--server.port", "8501"
            ])
            
            print("✅ Streamlit interface started!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def stop_services(self):
        """Stop all services"""
        print("\n🛑 Stopping services...")
        self.running = False
        
        if self.api_process:
            self.api_process.terminate()
            try:
                self.api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.api_process.kill()
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
        
        print("✅ All services stopped")
    
    def run(self):
        """Main execution function"""
        print("🧠 Trilingual AI System Launcher")
        print("=" * 40)
        
        # Check if required files exist
        required_files = [
            "multi_model_api.py",
            "streamlit_app.py", 
            "api_requirements.txt",
            "requirements.txt"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        
        try:
            # Start API server
            if not self.start_api():
                print("❌ Failed to start API server")
                return False
            
            # Start Streamlit interface
            if not self.start_streamlit():
                print("❌ Failed to start Streamlit interface")
                self.stop_services()
                return False
            
            # Show success message
            print("\n🎉 Trilingual AI System is running!")
            print("=" * 40)
            print("📖 API Documentation: http://localhost:8001/docs")
            print("💬 Chat Interface:    http://localhost:8501")
            print("🔍 API Health Check:  http://localhost:8001/health")
            print("\nPress Ctrl+C to stop all services")
            
            # Keep running until interrupted
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.stop_services()
        
        return True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown signal received...")
    launcher.stop_services()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the launcher
    launcher = TrilingualLauncher()
    success = launcher.run()
    
    sys.exit(0 if success else 1)
