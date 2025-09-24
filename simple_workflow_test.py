#!/usr/bin/env python3
"""
Simple workflow test without external dependencies
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append('.')

async def test_workflow_import():
    """Test if the workflow components can be imported"""
    try:
        print("🧪 Testing Workflow Import...")
        
        # Test basic imports
        from online_agent_service import OnlineAgent, OnlineWorkflowRequest, MessageType
        
        print("✅ Basic imports successful")
        
        # Test agent creation
        agent = OnlineAgent(
            id="test1",
            name="Test Agent",
            role="coordinator",
            model="mistral-small",
            system_prompt="You are a test coordinator"
        )
        print("✅ Agent creation successful")
        
        # Test workflow request creation
        request = OnlineWorkflowRequest(
            task="Test task",
            agents=[agent],
            conversation_id=None,
            enable_streaming=True
        )
        print("✅ Workflow request creation successful")
        
        print("🎉 All basic components working!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_import())
    if success:
        print("\n✅ Basic workflow functionality is working!")
    else:
        print("\n❌ Basic workflow functionality has issues")
