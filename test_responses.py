"""
Test script for multilingual responses
"""

import sys
sys.path.append('.')

from multi_model_api import MultiModelSystem

def test_responses():
    """Test language-specific responses"""
    print("🧪 Testing Multilingual Responses...")
    
    # Initialize system
    system = MultiModelSystem()
    
    # Test messages in different languages
    test_cases = [
        ("Hello, how are you?", "english"),
        ("Habari yako?", "kiswahili"), 
        ("Wĩ atĩa?", "kikuyu"),
        ("Inadi?", "luo"),
        ("What can you help me with?", "english"),
        ("Unaweza kunisaidia nini?", "kiswahili"),
        ("Ũngĩndeithagia atĩa?", "kikuyu"),
        ("Inyalo konya nadi?", "luo")
    ]
    
    print("\n📝 Testing responses:")
    for message, expected_lang in test_cases:
        print(f"\n🔹 Input: '{message}' (Expected: {expected_lang})")
        
        try:
            result = system.generate_response(
                message=message,
                language="auto",
                max_length=50,
                temperature=0.7
            )
            
            print(f"   🤖 Response: '{result['response']}'")
            print(f"   🌍 Detected: {result['language_detected']}")
            print(f"   🔧 Model: {result['model_used']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Language response testing completed!")

if __name__ == "__main__":
    test_responses()
