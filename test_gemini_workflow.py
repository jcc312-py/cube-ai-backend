#!/usr/bin/env python3
"""
Test workflow with Gemini model
"""

import requests
import json

def test_gemini_workflow():
    """Test the online workflow with Gemini model"""
    
    print("🔍 Testing Online Workflow with Gemini...")
    
    # Test request with Gemini
    test_request = {
        'task': 'Write a simple Python function to add two numbers',
        'agents': [{
            'id': 'coder',
            'name': 'Coder',
            'role': 'coder',
            'model': 'gemini-pro',
            'system_prompt': 'You are a Python developer',
            'memory_enabled': True
        }],
        'enable_streaming': True
    }
    
    print(f"📝 Task: {test_request['task']}")
    print(f"🤖 Agents: {len(test_request['agents'])}")
    print(f"🔧 Model: {test_request['agents'][0]['model']}")
    print()
    
    try:
        # Make the request
        response = requests.post('http://localhost:8000/run-online-workflow', json=test_request)
        result = response.json()
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"🆔 Workflow ID: {result.get('workflow_id')}")
        print(f"📊 Workflow Status: {result.get('status')}")
        print(f"🤖 Agent Status: {result.get('agents')}")
        print(f"📨 Total Messages: {result.get('total_messages')}")
        print()
        
        # Print message history
        message_history = result.get('message_history', [])
        print(f"📜 Message History ({len(message_history)} messages):")
        for i, msg in enumerate(message_history):
            print(f"  {i+1}. From: {msg.get('from_agent')} → To: {msg.get('to_agent')}")
            print(f"     Type: {msg.get('message_type')}")
            content = msg.get('content', '')
            print(f"     Content: {content[:200]}{'...' if len(content) > 200 else ''}")
            print()
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Gemini Workflow Test")
    print("=" * 50)
    print()
    
    test_gemini_workflow()
    
    print("🏁 Test complete!")
