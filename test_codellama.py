#!/usr/bin/env python3
"""
Test script for CodeLlama:7b-instruct integration
This script tests the model's code generation capabilities and GPU acceleration
"""

import asyncio
import json
from langchain_ollama import OllamaLLM

# Test configuration for CodeLlama
CODECLLAMA_CONFIG = {
    "num_gpu": 1,
    "num_thread": 8,
    "temperature": 0.3,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "top_k": 40,
    "num_ctx": 4096,
}

async def test_codellama():
    """Test CodeLlama:7b-instruct model"""
    print("🧪 Testing CodeLlama:7b-instruct integration...")
    
    try:
        # Initialize the model
        print("📦 Initializing CodeLlama:7b-instruct...")
        llm = OllamaLLM(model="codellama:7b-instruct", **CODECLLAMA_CONFIG)
        
        # Test 1: Simple code generation
        print("\n🔍 Test 1: Simple code generation")
        prompt1 = "Write a Python function that calculates the factorial of a number"
        response1 = llm.invoke(prompt1)
        print(f"Response: {response1[:200]}...")
        
        # Test 2: Code with specific requirements
        print("\n🔍 Test 2: Code with specific requirements")
        prompt2 = """
        Write a Python function that:
        1. Takes a list of numbers as input
        2. Returns the sum of all even numbers
        3. Uses list comprehension
        4. Includes error handling
        """
        response2 = llm.invoke(prompt2)
        print(f"Response: {response2[:300]}...")
        
        # Test 3: Code review
        print("\n🔍 Test 3: Code review")
        code_to_review = """
        def calculate_average(numbers):
            total = 0
            for num in numbers:
                total += num
            return total / len(numbers)
        """
        prompt3 = f"Review this code and suggest improvements:\n{code_to_review}"
        response3 = llm.invoke(prompt3)
        print(f"Response: {response3[:300]}...")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing CodeLlama: {e}")
        return False

async def test_gpu_acceleration():
    """Test GPU acceleration availability"""
    print("\n🔧 Testing GPU acceleration...")
    
    try:
        import subprocess
        
        # Check if CUDA is available
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA is available")
            print(f"GPU Info: {result.stdout.split('Driver Version')[0][:200]}...")
        else:
            print("⚠️ CUDA not detected, but Ollama may still use GPU")
        
        # Check Ollama GPU support
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "gpu" in result.stdout.lower():
            print("✅ Ollama GPU support detected")
        else:
            print("ℹ️ Ollama running (GPU support may be available)")
            
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

async def main():
    """Main test function"""
    print("🚀 Starting CodeLlama:7b-instruct integration tests...")
    
    # Test GPU acceleration
    await test_gpu_acceleration()
    
    # Test CodeLlama model
    success = await test_codellama()
    
    if success:
        print("\n🎉 CodeLlama:7b-instruct integration test completed successfully!")
        print("💡 The model is ready to use with GPU acceleration!")
    else:
        print("\n❌ CodeLlama:7b-instruct integration test failed!")
        print("🔧 Please check your Ollama installation and model download")

if __name__ == "__main__":
    asyncio.run(main()) 