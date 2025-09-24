#!/usr/bin/env python3
"""
Simple test script to verify the backend is working
"""

import requests
import json
import time

def test_backend():
    """Test the backend endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing backend endpoints...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test chat endpoint
    try:
        chat_data = {
            "prompt": "create a simple function that adds two numbers"
        }
        response = requests.post(f"{base_url}/chat", json=chat_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat endpoint passed: {result.get('type', 'unknown')}")
            if result.get('code'):
                print(f"📝 Code generated: {len(result['code'])} characters")
            if result.get('tests'):
                print(f"🧪 Tests generated: {len(result['tests'])} characters")
            if result.get('test_results'):
                print(f"🏃 Tests executed: {len(result['test_results'])} characters")
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False
    
    # Test file listing
    try:
        response = requests.get(f"{base_url}/list-files")
        if response.status_code == 200:
            files = response.json().get('files', [])
            print(f"✅ File listing passed: {len(files)} files found")
        else:
            print(f"❌ File listing failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ File listing error: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 Starting backend tests...")
    success = test_backend()
    if success:
        print("✅ Backend is working correctly!")
    else:
        print("❌ Backend has issues!") 