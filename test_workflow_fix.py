#!/usr/bin/env python3
"""
Test script for workflow routing fix
"""

import asyncio
import json
import requests
from datetime import datetime

async def test_workflow_routing():
    """Test the workflow routing with multiple agents"""
    
    # Test data
    test_request = {
        "task": "Create a simple Python calculator with addition and subtraction functions",
        "agents": [
            {
                "id": "coordinator1",
                "name": "Project Coordinator",
                "role": "coordinator",
                "model": "mistral-small",
                "system_prompt": "You are a project coordinator. Delegate tasks to appropriate agents and oversee the workflow."
            },
            {
                "id": "coder1", 
                "name": "Python Coder",
                "role": "coder",
                "model": "mistral-small",
                "system_prompt": "You are a Python developer. Write clean, functional code based on requirements."
            },
            {
                "id": "tester1",
                "name": "Code Tester", 
                "role": "tester",
                "model": "mistral-small",
                "system_prompt": "You are a code tester. Validate code quality and functionality."
            }
        ],
        "conversation_id": None,
        "enable_streaming": True
    }
    
    try:
        print("🧪 Testing Workflow Routing Fix...")
        print("=" * 60)
        
        # Test backend connection
        print("1. Testing backend connection...")
        response = requests.get("http://localhost:8000/online/health")
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend not responding")
            return False
        
        # Test workflow execution
        print("\n2. Testing workflow execution...")
        response = requests.post(
            "http://localhost:8000/online/run-workflow",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Workflow executed successfully")
            print(f"   Workflow ID: {result.get('workflow_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Total Messages: {result.get('total_messages')}")
            
            # Check message history
            messages = result.get('message_history', [])
            print(f"\n3. Message Flow Analysis:")
            print(f"   Total messages: {len(messages)}")
            
            for i, msg in enumerate(messages):
                print(f"   {i+1}. {msg.get('from_agent', 'unknown')} -> {msg.get('to_agent', 'unknown')}")
                print(f"      Type: {msg.get('message_type', 'unknown')}")
                content = msg.get('content', '')[:80]
                print(f"      Content: {content}...")
            
            # Check if multiple agents participated
            unique_agents = set()
            for msg in messages:
                if msg.get('from_agent') != 'system':
                    unique_agents.add(msg.get('from_agent'))
            
            print(f"\n4. Agent Participation:")
            print(f"   Agents involved: {list(unique_agents)}")
            
            if len(unique_agents) > 1:
                print("✅ Multiple agents participated - routing is working!")
                return True
            else:
                print("❌ Only one agent participated - routing issue detected")
                return False
                
        else:
            print(f"❌ Workflow execution failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_routing())
    if success:
        print("\n🎉 Workflow routing fix is working!")
    else:
        print("\n⚠️  Workflow routing still has issues")
