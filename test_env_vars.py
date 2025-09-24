#!/usr/bin/env python3
"""
Test environment variables in workflow context
"""

import requests
import json

def test_env_vars_endpoint():
    """Test if environment variables are accessible through the API"""
    
    print("🔍 Testing Environment Variables in API Context...")
    
    # Create a test request that will trigger environment variable usage
    test_request = {
        'task': 'Test environment variables',
        'agents': [{
            'id': 'test-agent',
            'name': 'Test Agent',
            'role': 'tester',
            'model': 'mistral-small',
            'system_prompt': 'You are a test agent',
            'memory_enabled': True
        }],
        'enable_streaming': True
    }
    
    try:
        response = requests.post('http://localhost:8000/run-online-workflow', json=test_request)
        result = response.json()
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📊 Workflow Status: {result.get('status')}")
        print(f"🤖 Agent Status: {result.get('agents')}")
        
        # Check message history for any error details
        message_history = result.get('message_history', [])
        for i, msg in enumerate(message_history):
            print(f"📨 Message {i+1}: From {msg.get('from_agent')} → To {msg.get('to_agent')}")
            content = msg.get('content', '')
            if 'Error:' in content:
                print(f"❌ Error found: {content}")
            else:
                print(f"✅ Content: {content[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_direct_env_check():
    """Test environment variables directly"""
    
    print("🔍 Testing Environment Variables Directly...")
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    mistral_key = os.getenv('MISTRAL_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    print(f"MISTRAL_API_KEY: {'✅ SET' if mistral_key else '❌ NOT SET'}")
    print(f"OPENAI_API_KEY: {'✅ SET' if openai_key else '❌ NOT SET'}")
    print(f"GEMINI_API_KEY: {'✅ SET' if gemini_key else '❌ NOT SET'}")
    
    if mistral_key:
        print(f"Mistral key length: {len(mistral_key)}")
        print(f"Mistral key starts with: {mistral_key[:10]}...")

if __name__ == "__main__":
    print("🚀 Environment Variables Debug Tool")
    print("=" * 50)
    print()
    
    test_direct_env_check()
    print()
    test_env_vars_endpoint()
    
    print("🏁 Debug complete!")
