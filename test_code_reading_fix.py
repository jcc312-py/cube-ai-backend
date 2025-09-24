#!/usr/bin/env python3
"""
Test script to demonstrate the code reading fix for agents
This script simulates the scenario where agents can now see code from pulled repositories
"""

import asyncio
import httpx
import json

async def test_code_reading():
    """Test the code reading functionality"""
    print("🧪 Testing Code Reading Fix for Agents")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: List available projects
        print("\n1️⃣ Testing project listing...")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/projects/list")
            if response.status_code == 200:
                data = response.json()
                projects = data.get("projects", [])
                print(f"✅ Found {len(projects)} projects")
                for project in projects[:3]:  # Show first 3
                    print(f"   - {project.get('name', 'Unknown')}: {project.get('description', 'No description')}")
            else:
                print(f"❌ Failed to list projects: {response.status_code}")
        
        # Test 2: Read code from projects
        print("\n2️⃣ Testing code reading from projects...")
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/agents/read-code", json={
                "include_generated": True,
                "file_pattern": "*.py"
            })
            if response.status_code == 200:
                data = response.json()
                code_files = data.get("code_files", [])
                print(f"✅ Found {len(code_files)} code files")
                
                # Show file details
                for file_info in code_files[:5]:  # Show first 5 files
                    print(f"   - {file_info['source']}/{file_info['file']} ({file_info['lines']} lines)")
                    if file_info['is_test']:
                        print(f"     🧪 Test file")
                
                if len(code_files) > 5:
                    print(f"   ... and {len(code_files) - 5} more files")
            else:
                print(f"❌ Failed to read code: {response.status_code}")
        
        # Test 3: Simulate agent conversation
        print("\n3️⃣ Testing agent conversation simulation...")
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/chat", json={
                "message": "can you see the code improve it let me know if you cant see anything"
            })
            if response.status_code == 200:
                data = response.json()
                print("✅ Agent response received:")
                print(f"   {data.get('response', 'No response')}")
            else:
                print(f"❌ Failed to get agent response: {response.status_code}")
        
        print("\n🎉 Code reading fix test completed!")
        print("\nThe agents should now be able to:")
        print("✅ See code from pulled git repositories")
        print("✅ Read code from project directories")
        print("✅ Access generated code files")
        print("✅ Provide meaningful responses about available code")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Code Reading Fix Test")
    print("Make sure the backend server is running on http://localhost:8000")
    print()
    
    asyncio.run(test_code_reading())
