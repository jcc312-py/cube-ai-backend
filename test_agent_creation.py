#!/usr/bin/env python3
"""
Test agent creation and message processing
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the online agent service components
from online_agent_service import OnlineAgent, OnlineAgentInstance, OnlineAgentMessage, MessageType

async def test_agent_creation():
    """Test creating and using an online agent"""
    
    print("🔍 Testing Agent Creation...")
    
    # Create agent config
    agent_config = OnlineAgent(
        id="test-coder",
        name="Test Coder",
        role="coder",
        model="mistral-small",
        system_prompt="You are a Python developer",
        memory_enabled=True
    )
    
    print(f"📝 Agent Config: {agent_config}")
    
    try:
        # Create agent instance
        print("🤖 Creating agent instance...")
        agent = OnlineAgentInstance(agent_config)
        print("✅ Agent instance created successfully")
        
        # Test message processing
        print("📨 Testing message processing...")
        test_message = OnlineAgentMessage(
            from_agent="system",
            to_agent="test-coder",
            message_type=MessageType.TASK,
            content="Write a simple Python function to add two numbers"
        )
        
        print(f"📝 Test Message: {test_message}")
        
        # Process message
        response = await agent.process_message(test_message)
        print(f"✅ Response received: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_creation():
    """Test LLM creation directly"""
    
    print("🔍 Testing LLM Creation...")
    
    try:
        from langchain_mistralai import ChatMistralAI
        
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            print("❌ MISTRAL_API_KEY not set")
            return False
        
        print("🤖 Creating Mistral LLM...")
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            max_tokens=4000,
            streaming=True,
            mistral_api_key=mistral_key
        )
        
        print("✅ Mistral LLM created successfully")
        
        # Test a simple generation
        print("📨 Testing LLM generation...")
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content="Write a simple Python function to add two numbers")]
        response = await llm.agenerate([messages])
        
        result = response.generations[0][0].text
        print(f"✅ LLM Response: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Creation Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Agent Creation Debug Tool")
    print("=" * 50)
    print()
    
    # Test API keys
    print("🔑 Checking API Keys...")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    print(f"MISTRAL_API_KEY: {'✅ SET' if mistral_key else '❌ NOT SET'}")
    print()
    
    # Test LLM creation
    llm_success = await test_llm_creation()
    print()
    
    if llm_success:
        # Test agent creation
        agent_success = await test_agent_creation()
        print()
        
        if agent_success:
            print("🎉 All tests passed!")
        else:
            print("❌ Agent creation failed")
    else:
        print("❌ LLM creation failed - skipping agent test")
    
    print("🏁 Debug complete!")

if __name__ == "__main__":
    asyncio.run(main())
