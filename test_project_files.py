#!/usr/bin/env python3
"""
Test script to verify that pulled project files are accessible
"""

import asyncio
import httpx
import json
from pathlib import Path

async def test_project_file_access():
    """Test that we can access files from pulled projects"""
    print("🧪 Testing Project File Access")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    try:
        # Test 1: List all projects
        print("\n1️⃣ Listing all projects...")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/projects/list")
            if response.status_code == 200:
                data = response.json()
                projects = data.get("projects", [])
                print(f"✅ Found {len(projects)} projects")
                
                for project in projects:
                    print(f"   📁 {project['name']} ({project.get('source', 'unknown')}) - {project.get('file_count', 0)} files")
                    print(f"      Type: {project.get('type', 'unknown')}")
                    print(f"      Path: {project.get('path', 'unknown')}")
            else:
                print(f"❌ Failed to list projects: {response.status_code}")
                return
        
        # Test 2: Check if projects directory exists locally
        print("\n2️⃣ Checking local projects directory...")
        projects_dir = Path("generated/projects")
        if projects_dir.exists():
            print(f"✅ Projects directory exists: {projects_dir}")
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    file_count = len(list(project_dir.rglob("*")))
                    print(f"   📁 {project_dir.name}: {file_count} files")
                    
                    # Show some files
                    files = list(project_dir.rglob("*"))[:5]
                    for file_path in files:
                        if file_path.is_file():
                            print(f"      📄 {file_path.relative_to(project_dir)}")
        else:
            print(f"❌ Projects directory not found: {projects_dir}")
        
        # Test 3: Try to read code from each project
        print("\n3️⃣ Testing code reading from each project...")
        for project in projects:
            project_name = project['name']
            print(f"\n   Testing project: {project_name}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/projects/{project_name}/code")
                if response.status_code == 200:
                    data = response.json()
                    files = data.get("files", [])
                    print(f"   ✅ Found {len(files)} files in {project_name}")
                    
                    # Show file details
                    for file_info in files[:3]:  # Show first 3 files
                        print(f"      📄 {file_info['name']} ({file_info['lines']} lines)")
                        if file_info['is_test']:
                            print(f"         🧪 Test file")
                        # Show first few lines of content
                        content_preview = file_info['content'][:200]
                        print(f"         Preview: {content_preview}...")
                    
                    if len(files) > 3:
                        print(f"      ... and {len(files) - 3} more files")
                else:
                    print(f"   ❌ Failed to read code from {project_name}: {response.status_code}")
                    print(f"      Error: {response.text}")
        
        # Test 4: Test agent code reading
        print("\n4️⃣ Testing agent code reading...")
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/agents/read-code", json={
                "include_generated": True,
                "file_pattern": "*"
            })
            if response.status_code == 200:
                data = response.json()
                code_files = data.get("code_files", [])
                print(f"✅ Agent code reading found {len(code_files)} files")
                
                # Group by project
                by_project = {}
                for file_info in code_files:
                    project = file_info.get('project', 'unknown')
                    if project not in by_project:
                        by_project[project] = []
                    by_project[project].append(file_info)
                
                for project, files in by_project.items():
                    print(f"   📁 {project}: {len(files)} files")
            else:
                print(f"❌ Agent code reading failed: {response.status_code}")
        
        print("\n🎉 Project file access test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Project File Access Test")
    print("Make sure the backend server is running on http://localhost:8000")
    print()
    
    asyncio.run(test_project_file_access())
