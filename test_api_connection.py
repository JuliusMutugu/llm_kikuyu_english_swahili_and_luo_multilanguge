#!/usr/bin/env python3
"""
Test API Connection
Quick script to test if the multi_model_api.py is running and accessible
"""

import requests
import json

def test_api_connection():
    """Test connection to the API server"""
    ports = [8001, 8000, 8002, 8003, 3000, 5000]
    
    print("🔍 Testing API connection on multiple ports...")
    
    for port in ports:
        try:
            print(f"\n📡 Testing port {port}...")
            
            # Test health endpoint
            health_url = f'http://localhost:{port}/health'
            print(f"   Trying: {health_url}")
            
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ API server found on port {port}!")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
                
                # Test chat endpoint
                chat_url = f'http://localhost:{port}/chat'
                test_message = {
                    "message": "Hello, test message",
                    "language": "english"
                }
                
                print(f"\n💬 Testing chat endpoint...")
                chat_response = requests.post(chat_url, json=test_message, timeout=10)
                
                if chat_response.status_code == 200:
                    print(f"✅ Chat endpoint working!")
                    chat_data = chat_response.json()
                    print(f"   Response: {chat_data.get('response', 'No response')[:100]}...")
                    print(f"   Language: {chat_data.get('language_detected', 'Unknown')}")
                    print(f"   Confidence: {chat_data.get('confidence', 0):.2%}")
                else:
                    print(f"❌ Chat endpoint failed: {chat_response.status_code}")
                    print(f"   Error: {chat_response.text}")
                
                return port
                
            else:
                print(f"   ❌ Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection refused - server not running on port {port}")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout - server might be slow on port {port}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n❌ No API server found on any tested ports")
    print(f"💡 Make sure to start the API server with: python multi_model_api.py")
    return None

if __name__ == "__main__":
    found_port = test_api_connection()
    
    if found_port:
        print(f"\n🎉 Success! API server is running on port {found_port}")
        print(f"🔗 Use this URL in your Streamlit app: http://localhost:{found_port}")
    else:
        print(f"\n💔 No API server found. Please check:")
        print(f"   1. Is multi_model_api.py running?")
        print(f"   2. Check the terminal output for any errors")
        print(f"   3. Try restarting the API server")
