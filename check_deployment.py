#!/usr/bin/env python3
"""
🔍 Render Deployment Status Checker
Check the health and status of your deployed services on Render.
"""

import requests
import time
import json
from datetime import datetime

class RenderStatusChecker:
    def __init__(self):
        self.api_url = None
        self.ui_url = None
        
    def set_urls(self, api_url=None, ui_url=None):
        """Set the URLs for your deployed services"""
        self.api_url = api_url or input("Enter your API URL (e.g., https://trilingual-ai-api.onrender.com): ").strip()
        self.ui_url = ui_url or input("Enter your UI URL (e.g., https://trilingual-ai-ui.onrender.com): ").strip()
        
    def check_api_health(self):
        """Check if API service is healthy"""
        print(f"\n🔍 Checking API Health: {self.api_url}")
        try:
            # Check health endpoint
            response = requests.get(f"{self.api_url}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Health: {data.get('status', 'unknown')}")
                print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
                return True
            else:
                print(f"❌ API Health Check Failed: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ API Health Check Failed: {str(e)}")
            return False
    
    def check_api_docs(self):
        """Check if API docs are accessible"""
        print(f"\n📚 Checking API Docs: {self.api_url}/docs")
        try:
            response = requests.get(f"{self.api_url}/docs", timeout=15)
            if response.status_code == 200:
                print("✅ API Docs: Accessible")
                return True
            else:
                print(f"❌ API Docs: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ API Docs Failed: {str(e)}")
            return False
    
    def test_chat_endpoint(self):
        """Test the chat functionality"""
        print(f"\n💬 Testing Chat Endpoint")
        try:
            test_data = {
                "message": "Hello, how are you?",
                "language": "english"
            }
            response = requests.post(
                f"{self.api_url}/chat", 
                json=test_data, 
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                print("✅ Chat Endpoint: Working")
                print(f"   Response: {data.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Chat Endpoint: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Chat Endpoint Failed: {str(e)}")
            return False
    
    def check_ui_accessibility(self):
        """Check if UI is accessible"""
        print(f"\n🖥️ Checking UI Accessibility: {self.ui_url}")
        try:
            response = requests.get(self.ui_url, timeout=30)
            if response.status_code == 200:
                print("✅ UI: Accessible")
                return True
            else:
                print(f"❌ UI: Status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ UI Failed: {str(e)}")
            return False
    
    def test_response_times(self):
        """Test response times for critical endpoints"""
        print(f"\n⏱️ Testing Response Times")
        
        endpoints = [
            ("/health", "Health Check"),
            ("/", "Root Endpoint"),
            ("/docs", "API Docs")
        ]
        
        for endpoint, name in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}{endpoint}", timeout=15)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    if response_time < 1000:
                        print(f"✅ {name}: {response_time:.0f}ms (Good)")
                    elif response_time < 3000:
                        print(f"⚠️ {name}: {response_time:.0f}ms (Slow)")
                    else:
                        print(f"❌ {name}: {response_time:.0f}ms (Very Slow)")
                else:
                    print(f"❌ {name}: Status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {name}: Failed - {str(e)}")
    
    def comprehensive_check(self):
        """Run all checks in sequence"""
        print("🚀 Render Deployment Status Check")
        print("=" * 50)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.api_url or not self.ui_url:
            self.set_urls()
        
        # Run all checks
        checks = []
        checks.append(("API Health", self.check_api_health()))
        checks.append(("API Docs", self.check_api_docs()))
        checks.append(("Chat Functionality", self.test_chat_endpoint()))
        checks.append(("UI Accessibility", self.check_ui_accessibility()))
        
        # Response time test
        self.test_response_times()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 DEPLOYMENT STATUS SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for _, status in checks if status)
        total = len(checks)
        
        for check_name, status in checks:
            emoji = "✅" if status else "❌"
            print(f"{emoji} {check_name}")
        
        print(f"\n🎯 Overall Score: {passed}/{total} ({passed/total*100:.0f}%)")
        
        if passed == total:
            print("🎉 All systems are GO! Your deployment is healthy.")
        elif passed >= total * 0.75:
            print("⚠️ Most systems working, minor issues detected.")
        else:
            print("🚨 Major issues detected. Check logs and troubleshoot.")
        
        print(f"\n🔗 Your Services:")
        print(f"   API: {self.api_url}")
        print(f"   UI: {self.ui_url}")
        
        return passed == total

def main():
    checker = RenderStatusChecker()
    
    print("🔍 Render Deployment Status Checker")
    print("This tool will check the health of your deployed services.\n")
    
    # Option to provide URLs or use defaults
    use_defaults = input("Use default URLs? (y/n): ").lower().strip()
    
    if use_defaults == 'y':
        api_url = "https://trilingual-ai-api.onrender.com"
        ui_url = "https://trilingual-ai-ui.onrender.com"
        checker.set_urls(api_url, ui_url)
    else:
        checker.set_urls()
    
    # Run comprehensive check
    success = checker.comprehensive_check()
    
    # Option for continuous monitoring
    monitor = input("\nEnable continuous monitoring? (y/n): ").lower().strip()
    
    if monitor == 'y':
        interval = int(input("Check interval in minutes (default 5): ") or "5")
        print(f"\n🔄 Starting continuous monitoring (every {interval} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(interval * 60)
                print(f"\n⏰ Scheduled check - {datetime.now().strftime('%H:%M:%S')}")
                checker.comprehensive_check()
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()
