#!/usr/bin/env python3
"""
Test script for Gemini integration
"""

import os
import sys

def test_gemini_import():
    """Test if Gemini can be imported"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ Gemini import successful")
        return True
    except ImportError as e:
        print(f"❌ Gemini import failed: {e}")
        return False

def test_gemini_api_key():
    """Test if Gemini API key is available"""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ Gemini API key found: {api_key[:10]}...")
        return True
    else:
        print("❌ Gemini API key not found")
        return False

def test_gemini_llm_creation():
    """Test if Gemini LLM can be created"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Cannot test LLM creation without API key")
            return False
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            max_tokens=1000,
            google_api_key=api_key
        )
        print("✅ Gemini LLM creation successful")
        return True
    except Exception as e:
        print(f"❌ Gemini LLM creation failed: {e}")
        return False

def main():
    """Run all Gemini tests"""
    print("🧪 Testing Gemini Integration...")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_gemini_import),
        ("API Key Test", test_gemini_api_key),
        ("LLM Creation Test", test_gemini_llm_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All Gemini tests passed! Integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
