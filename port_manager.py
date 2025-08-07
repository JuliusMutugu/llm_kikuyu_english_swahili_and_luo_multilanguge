"""
Port Management Utility
Check and manage port usage for the API server
"""

import subprocess
import sys
import socket
import time

def check_port(port):
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True  # Port is available
        except OSError:
            return False  # Port is in use

def kill_port_process(port):
    """Kill process using the specified port (Windows)"""
    try:
        # Find process using the port
        result = subprocess.run(
            ['netstat', '-ano', '|', 'findstr', f':{port}'],
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        print(f"🔍 Found process {pid} using port {port}")
                        
                        # Kill the process
                        kill_result = subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True, text=True)
                        if kill_result.returncode == 0:
                            print(f"✅ Killed process {pid}")
                            return True
                        else:
                            print(f"❌ Failed to kill process {pid}")
                            return False
        
        print(f"🔍 No process found using port {port}")
        return True
        
    except Exception as e:
        print(f"❌ Error managing port {port}: {e}")
        return False

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if check_port(port):
            return port
    return None

def main():
    """Main port management function"""
    print("🔧 Port Management Utility")
    print("=" * 30)
    
    # Check port 8000
    if check_port(8000):
        print("✅ Port 8000 is available")
    else:
        print("❌ Port 8000 is in use")
        
        choice = input("🤔 Kill process on port 8000? (y/n): ").lower()
        if choice == 'y':
            if kill_port_process(8000):
                print("✅ Port 8000 is now available")
            else:
                print("❌ Could not free port 8000")
                
                # Find alternative port
                alt_port = find_available_port(8001)
                if alt_port:
                    print(f"💡 Alternative port available: {alt_port}")
                else:
                    print("❌ No alternative ports found")
    
    # Check other common ports
    common_ports = [8001, 8002, 8501, 8502]
    print("\n🔍 Checking other common ports:")
    for port in common_ports:
        status = "✅ Available" if check_port(port) else "❌ In use"
        print(f"   Port {port}: {status}")

if __name__ == "__main__":
    main()
