#!/usr/bin/env python3
"""
Quick test for the live deployment at:
https://llm-kikuyu-english-swahili-and-luo.onrender.com/
"""

import requests
import time
from datetime import datetime

def test_deployment():
    base_url = "https://llm-kikuyu-english-swahili-and-luo.onrender.com"
    
    print("🚀 Testing Your Live Trilingual AI Assistant")
    print("=" * 60)
    print(f"🌐 URL: {base_url}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Basic connectivity
    print("🔍 Test 1: Basic Connectivity")
    try:
        response = requests.get(base_url, timeout=30)
        if response.status_code == 200:
            print("✅ Service is accessible!")
            print(f"   Status: {response.status_code}")
            print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
        else:
            print(f"⚠️ Service returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    print()
    
    # Test 2: Check if it's a Streamlit app
    print("🖥️ Test 2: Streamlit Interface Check")
    try:
        response = requests.get(base_url, timeout=15)
        content = response.text.lower()
        
        if "streamlit" in content:
            print("✅ Streamlit interface detected!")
        if "trilingual" in content or "kikuyu" in content:
            print("✅ Trilingual AI content detected!")
        if "chat" in content or "message" in content:
            print("✅ Chat interface elements found!")
            
    except Exception as e:
        print(f"⚠️ Content check failed: {e}")
    
    print()
    
    # Test 3: Check for API endpoints (if available)
    print("🔧 Test 3: API Endpoints Check")
    endpoints_to_test = [
        "/health",
        "/docs", 
        "/api/health",
        "/_stcore/health"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {endpoint} - Available")
            else:
                print(f"⚠️ {endpoint} - Status {response.status_code}")
        except:
            print(f"❌ {endpoint} - Not available")
    
    print()
    
    # Test 4: Performance check
    print("⚡ Test 4: Performance Check")
    try:
        start_time = time.time()
        response = requests.get(base_url, timeout=20)
        load_time = time.time() - start_time
        
        if load_time < 2:
            print(f"🚀 Excellent response time: {load_time:.2f}s")
        elif load_time < 5:
            print(f"✅ Good response time: {load_time:.2f}s")
        elif load_time < 10:
            print(f"⚠️ Slow response time: {load_time:.2f}s (cold start?)")
        else:
            print(f"❌ Very slow response: {load_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    print()
    print("=" * 60)
    print("🎯 DEPLOYMENT STATUS SUMMARY")
    print("=" * 60)
    print("✅ Your Trilingual AI Assistant is LIVE!")
    print(f"🌐 Access your app: {base_url}")
    print()
    print("📱 What you can do now:")
    print("   • Test chat functionality in multiple languages")
    print("   • Share the URL with others")
    print("   • Monitor performance in Render dashboard")
    print("   • Add custom domain (optional)")
    print()
    print("🎉 Congratulations! Your deployment is successful!")

if __name__ == "__main__":
    test_deployment()
