"""
Simple API Server Launcher with Port Management
"""

import subprocess
import socket
import time
import sys
import os

def find_free_port(start_port=8000):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    return None

def start_api_server():
    """Start the API server"""
    print("🔍 Looking for available port...")
    
    # Find available port
    port = find_free_port(8000)
    if port is None:
        print("❌ No available ports found!")
        return False
    
    print(f"✅ Found available port: {port}")
    
    # Update the API file to use this specific port
    api_file = "multi_model_api.py"
    
    # Read the current file
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the uvicorn.run line to use our specific port
    import re
    pattern = r'uvicorn\.run\(app, host="0\.0\.0\.0", port=available_port\)'
    replacement = f'uvicorn.run(app, host="0.0.0.0", port={port})'
    
    updated_content = re.sub(pattern, replacement, content)
    
    # Write the updated content to a temporary file
    temp_api_file = f"temp_api_{port}.py"
    with open(temp_api_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"🚀 Starting API server on port {port}...")
    
    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, temp_api_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a moment for startup
        time.sleep(2)
        
        # Check if it's running
        if process.poll() is None:
            print(f"✅ API server started successfully!")
            print(f"📚 Documentation: http://localhost:{port}/docs")
            print(f"🌐 Chat endpoint: http://localhost:{port}/chat")
            print("\n" + "="*50)
            print("Press Ctrl+C to stop the server")
            print("="*50)
            
            try:
                # Keep the process running
                while True:
                    time.sleep(1)
                    if process.poll() is not None:
                        break
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                process.terminate()
                process.wait()
                
            # Clean up temp file
            try:
                os.remove(temp_api_file)
            except:
                pass
                
            return True
        else:
            # Process failed to start
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start server")
            print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🌍 MULTILINGUAL CHAT API LAUNCHER")
    print("=" * 60)
    
    success = start_api_server()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("  - Check if another service is using ports 8000-8020")
        print("  - Run: python port_manager.py")
        print("  - Try restarting your terminal")
