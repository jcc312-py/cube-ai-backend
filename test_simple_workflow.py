#!/usr/bin/env python3
"""
Simple workflow test to check basic functionality
"""

import requests
import json

def test_simple_workflow():
    """Test a simple workflow with coordinator and coder"""
    
    # Test data
    test_request = {
        "task": "Create a simple calculator function",
        "agents": [
            {
                "id": "coordinator1",
                "name": "Project Coordinator",
                "role": "coordinator",
                "model": "mistral-small",
                "system_prompt": "You are a project coordinator. Create a plan and delegate tasks."
            },
            {
                "id": "coder1", 
                "name": "Python Coder",
                "role": "coder",
                "model": "mistral-small",
                "system_prompt": "You are a Python developer. Write clean, functional code."
            }
        ],
        "conversation_id": None,
        "enable_streaming": True
    }
    
    try:
        print("🧪 Testing Simple Workflow...")
        print("=" * 50)
        
        # Test workflow execution
        print("1. Sending workflow request...")
        response = requests.post(
            "http://localhost:8000/online/run-workflow",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Workflow executed successfully!")
            print(f"   Workflow ID: {result.get('workflow_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Total Messages: {result.get('total_messages')}")
            
            # Check message history
            messages = result.get('message_history', [])
            print(f"\n2. Message Flow:")
            print(f"   Total messages: {len(messages)}")
            
            for i, msg in enumerate(messages):
                print(f"   {i+1}. {msg.get('from_agent', 'unknown')} -> {msg.get('to_agent', 'unknown')}")
                content = msg.get('content', '')[:100]
                print(f"      Content: {content}...")
            
            # Check if multiple agents participated
            unique_agents = set()
            for msg in messages:
                if msg.get('from_agent') != 'system':
                    unique_agents.add(msg.get('from_agent'))
            
            print(f"\n3. Agent Participation:")
            print(f"   Agents involved: {list(unique_agents)}")
            
            if len(unique_agents) > 1:
                print("✅ Multiple agents participated - workflow is working!")
                return True
            else:
                print("❌ Only one agent participated - workflow issue detected")
                return False
                
        else:
            print(f"❌ Workflow execution failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_simple_workflow()
    if success:
        print("\n🎉 Simple workflow test passed!")
    else:
        print("\n⚠️  Simple workflow test failed")
