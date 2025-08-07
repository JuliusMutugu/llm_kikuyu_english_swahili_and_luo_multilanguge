#!/usr/bin/env python3
"""
Test how the Streamlit app detects and uses API URLs
"""

import requests
import json

def test_api_endpoints():
    base_url = "https://llm-kikuyu-english-swahili-and-luo.onrender.com"
    
    print("🧪 Testing API Endpoints on Your Live Deployment")
    print("=" * 60)
    
    # Test 1: Health endpoint
    print("🔍 Test 1: Health Check")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Test 2: API Documentation
    print("📚 Test 2: API Documentation")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API docs are accessible")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Test 3: Chat endpoint
    print("💬 Test 3: Chat Functionality")
    
    test_messages = [
        {"message": "Hello, how are you?", "language": "english"},
        {"message": "Habari yako?", "language": "kiswahili"},
        {"message": "Wathii atia?", "language": "kikuyu"}
    ]
    
    for i, test_data in enumerate(test_messages, 1):
        try:
            print(f"Test 3.{i}: {test_data['language'].title()} - '{test_data['message']}'")
            response = requests.post(
                f"{base_url}/chat", 
                json=test_data, 
                timeout=15,
                headers={'Content-Type': 'application/json'}
            )
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data.get('response', 'No response')[:100]}...")
            else:
                print(f"Error response: {response.text[:200]}")
            print()
            
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("=" * 60)
    print("🎯 Summary: Your Streamlit app connects to these endpoints")
    print(f"🌐 Base URL: {base_url}")
    print("📡 Available endpoints:")
    print("   • /health - API health check")
    print("   • /docs - Interactive API documentation")
    print("   • /chat - Main chat functionality")
    print("   • / - Streamlit UI interface")

if __name__ == "__main__":
    test_api_endpoints()
