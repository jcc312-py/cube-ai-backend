#!/usr/bin/env python3
"""
Test if server is running and projects endpoint works
"""

import requests
import json

def test_server():
    """Test server connectivity and projects endpoint"""
    print("🧪 Testing Server and Projects Endpoint")
    print("=" * 50)
    
    # Test 1: Health check
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Server not running: {e}")
        return
    
    # Test 2: Projects endpoint
    try:
        response = requests.get('http://localhost:8000/projects', timeout=5)
        print(f"Projects endpoint: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Count: {data.get('count')}")
            
            projects = data.get('projects', [])
            print(f"Projects found: {len(projects)}")
            
            for project in projects:
                print(f"  - {project.get('name')} ({project.get('file_count', 0)} files)")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Projects endpoint failed: {e}")

if __name__ == "__main__":
    test_server()

