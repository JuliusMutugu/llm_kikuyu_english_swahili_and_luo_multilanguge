#!/usr/bin/env python3
"""
Test client for the Trilingual LLM API
"""

import requests
import json
import time
from typing import Dict, Any

class TrilingualLLMClient:
    """Client for interacting with the Trilingual LLM API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def chat(self, message: str, language: str = "auto", **kwargs) -> Dict[str, Any]:
        """Send a chat message"""
        payload = {
            "message": message,
            "language": language,
            **kwargs
        }
        
        try:
            response = self.session.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_languages(self) -> Dict[str, Any]:
        """Get supported languages"""
        try:
            response = self.session.get(f"{self.base_url}/languages")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def test_multilingual_conversations():
    """Test conversations in different languages"""
    client = TrilingualLLMClient()
    
    print("🚀 Testing Trilingual LLM API")
    print("=" * 50)
    
    # Health check
    health = client.health_check()
    print(f"🔍 Health Check: {health}")
    
    # Get model info
    info = client.get_info()
    if "error" not in info:
        print(f"🤖 Model: {info['name']}")
        print(f"📊 Parameters: {info['parameters']:,}")
        print(f"🌍 Languages: {', '.join(info['languages'])}")
    
    print("\n💬 Testing Conversations:")
    print("-" * 30)
    
    # Test messages in different languages
    test_messages = [
        ("Habari yako? Nakupenda sana!", "swahili"),
        ("Warikia atĩa? Nĩngũkwenda mũno!", "kikuyu"), 
        ("Nadi? Aheri miwuoro matek!", "luo"),
        ("Hello, how are you today?", "english"),
        ("Unafanya nini leo?", "auto"),  # Auto-detect
        ("Familia yangu ni muhimu", "auto"),
        ("Dala mar jo-Luo ber ahinya", "auto"),
    ]
    
    for i, (message, lang) in enumerate(test_messages, 1):
        print(f"\n{i}. 👤 User ({lang}): {message}")
        
        response = client.chat(message, language=lang)
        
        if "error" in response:
            print(f"   ❌ Error: {response['error']}")
        else:
            print(f"   🤖 Bot ({response['language_detected']}): {response['response']}")
            print(f"   📊 Confidence: {response['confidence']:.2f}")
            print(f"   🔢 Tokens: {response['tokens_generated']}")
        
        time.sleep(1)  # Be nice to the API
    
    print("\n🌍 Supported Languages:")
    languages = client.get_languages()
    if "error" not in languages:
        for lang in languages['languages']:
            print(f"   {lang['code']}: {lang['name']} ({lang['native_name']})")
            print(f"      Example: {lang['example']}")

if __name__ == "__main__":
    test_multilingual_conversations()
