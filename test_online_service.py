#!/usr/bin/env python3
"""
Test script for Online Agent Service
Tests the basic functionality without requiring API keys
"""

import asyncio
import json
from datetime import datetime
from online_agent_service import (
    OnlineAgent, 
    OnlineWorkflowRequest, 
    OnlineWorkflowManager,
    MessageType,
    OnlineAgentMessage
)

async def test_online_service():
    """Test the online agent service functionality"""
    print("🧪 Testing Online Agent Service...")
    
    # Test 1: Create agents
    print("\n1. Testing agent creation...")
    agents = [
        OnlineAgent(
            id="coordinator",
            name="Workflow Coordinator",
            role="coordinator",
            model="gpt-3.5-turbo",
            system_prompt="You are a workflow coordinator that manages tasks and delegates work to other agents."
        ),
        OnlineAgent(
            id="coder",
            name="Code Generator",
            role="programmer",
            model="gpt-3.5-turbo",
            system_prompt="You are a skilled programmer that writes clean, efficient code."
        ),
        OnlineAgent(
            id="reviewer",
            name="Code Reviewer",
            role="reviewer",
            model="mistral-medium",
            system_prompt="You are a code reviewer that checks code quality and provides feedback."
        )
    ]
    
    print(f"✅ Created {len(agents)} agents")
    
    # Test 2: Create workflow request
    print("\n2. Testing workflow request creation...")
    workflow_request = OnlineWorkflowRequest(
        task="Create a simple Python function that calculates the factorial of a number",
        agents=agents,
        enable_streaming=False
    )
    
    print(f"✅ Created workflow request with task: {workflow_request.task}")
    
    # Test 3: Test message creation
    print("\n3. Testing message creation...")
    test_message = OnlineAgentMessage(
        from_agent="system",
        to_agent="coordinator",
        message_type=MessageType.TASK,
        content="Test message content",
        metadata={"test": True}
    )
    
    print(f"✅ Created test message: {test_message.id}")
    
    # Test 4: Test workflow manager initialization
    print("\n4. Testing workflow manager...")
    try:
        workflow_manager = OnlineWorkflowManager()
        print("✅ Workflow manager initialized successfully")
        
        # Test 5: Test model configurations
        print("\n5. Testing model configurations...")
        from online_agent_service import ONLINE_MODEL_CONFIGS, DEFAULT_ONLINE_MODEL
        print(f"✅ Available models: {list(ONLINE_MODEL_CONFIGS.keys())}")
        print(f"✅ Default model: {DEFAULT_ONLINE_MODEL}")
        
        # Test 6: Test API endpoints (without running server)
        print("\n6. Testing API structure...")
        from online_agent_service import online_app
        routes = [route.path for route in online_app.routes]
        print(f"✅ Available endpoints: {routes}")
        
    except Exception as e:
        print(f"❌ Error initializing workflow manager: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    print("\n📋 Service Summary:")
    print(f"   • Available models: {len(ONLINE_MODEL_CONFIGS)}")
    print(f"   • OpenAI models: {[k for k, v in ONLINE_MODEL_CONFIGS.items() if v['provider'] == 'openai']}")
    print(f"   • Mistral models: {[k for k, v in ONLINE_MODEL_CONFIGS.items() if v['provider'] == 'mistral']}")
    print(f"   • API endpoints: {len(routes)}")
    print(f"   • Default model: {DEFAULT_ONLINE_MODEL}")
    
    return True

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from online_agent_service import (
            OnlineAgent, OnlineWorkflowRequest, OnlineWorkflowManager,
            MessageType, OnlineAgentMessage, OnlineAgentStatus,
            ONLINE_MODEL_CONFIGS, DEFAULT_ONLINE_MODEL
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Online Agent Service Test Suite")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed")
        exit(1)
    
    # Test service functionality
    success = asyncio.run(test_online_service())
    
    if success:
        print("\n✅ All tests passed! The service is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Set your API keys:")
        print("      - OPENAI_API_KEY for OpenAI models")
        print("      - MISTRAL_API_KEY for Mistral models")
        print("   2. Run the service: python online_agent_service.py")
        print("   3. Access API docs: http://localhost:8001/docs")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
