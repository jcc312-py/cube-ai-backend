#!/usr/bin/env python3
"""
Debug script for online workflow issues
"""

import requests
import json
import os

def test_workflow():
    """Test the online workflow with detailed logging"""
    
    # Test request
    test_request = {
        'task': 'Write a simple Python function to add two numbers',
        'agents': [{
            'id': 'coder',
            'name': 'Coder',
            'role': 'coder',
            'model': 'mistral-small',
            'system_prompt': 'You are a Python developer',
            'memory_enabled': True
        }],
        'enable_streaming': True
    }
    
    print("🔍 Testing Online Workflow...")
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
        
        # Check for errors
        if result.get('status') == 'error':
            print("❌ Workflow failed!")
            print("🔍 Checking agent statuses...")
            for agent_id, status in result.get('agents', {}).items():
                print(f"  - {agent_id}: {status}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_api_keys():
    """Test if API keys are properly set"""
    print("🔑 Checking API Keys...")
    
    keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'MISTRAL_API_KEY': os.getenv('MISTRAL_API_KEY'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')
    }
    
    for key, value in keys.items():
        status = "✅ SET" if value else "❌ NOT SET"
        print(f"  {key}: {status}")
    
    print()

def test_models_endpoint():
    """Test the models endpoint"""
    print("🔍 Testing Models Endpoint...")
    
    try:
        response = requests.get('http://localhost:8000/online-models')
        if response.status_code == 200:
            data = response.json()
            print("✅ Models endpoint working")
            print(f"📊 Available models: {len(data.get('available_models', {}))}")
            for model_name, config in data.get('available_models', {}).items():
                print(f"  - {model_name}: {config.get('provider', 'unknown')}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
    
    print()

if __name__ == "__main__":
    print("🚀 Online Workflow Debug Tool")
    print("=" * 50)
    print()
    
    test_api_keys()
    test_models_endpoint()
    test_workflow()
    
    print("🏁 Debug complete!")
