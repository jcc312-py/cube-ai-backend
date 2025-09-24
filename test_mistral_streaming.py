#!/usr/bin/env python3
"""
Test Mistral API with and without streaming
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_mistral_with_streaming():
    """Test Mistral with streaming enabled"""
    print("🔍 Testing Mistral with streaming...")
    
    try:
        from langchain_mistralai import ChatMistralAI
        
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            print("❌ MISTRAL_API_KEY not set")
            return False
        
        # Test with streaming enabled
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            max_tokens=4000,
            streaming=True,
            mistral_api_key=mistral_key
        )
        
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Write a simple Python function to add two numbers")]
        
        print("📨 Testing with streaming=True...")
        response = await llm.agenerate([messages])
        result = response.generations[0][0].text
        print(f"✅ Streaming Response: {result[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Streaming Error: {e}")
        return False

async def test_mistral_without_streaming():
    """Test Mistral with streaming disabled"""
    print("🔍 Testing Mistral without streaming...")
    
    try:
        from langchain_mistralai import ChatMistralAI
        
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            print("❌ MISTRAL_API_KEY not set")
            return False
        
        # Test with streaming disabled
        llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.3,
            max_tokens=4000,
            streaming=False,
            mistral_api_key=mistral_key
        )
        
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Write a simple Python function to add two numbers")]
        
        print("📨 Testing with streaming=False...")
        response = await llm.agenerate([messages])
        result = response.generations[0][0].text
        print(f"✅ Non-streaming Response: {result[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Non-streaming Error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Mistral Streaming Debug Tool")
    print("=" * 50)
    print()
    
    # Test with streaming
    streaming_success = await test_mistral_with_streaming()
    print()
    
    # Test without streaming
    non_streaming_success = await test_mistral_without_streaming()
    print()
    
    if streaming_success and non_streaming_success:
        print("🎉 Both streaming and non-streaming work!")
    elif non_streaming_success:
        print("⚠️ Only non-streaming works - streaming has issues")
    else:
        print("❌ Both streaming and non-streaming failed")
    
    print("🏁 Debug complete!")

if __name__ == "__main__":
    asyncio.run(main())
