#!/usr/bin/env python3
"""
Final test of the projects API
"""

import requests
import json

def test_final():
    """Test the projects API one more time"""
    print("🧪 Final Test of Projects API")
    print("=" * 40)
    
    try:
        # Test health first
        health = requests.get('http://localhost:8000/health', timeout=5)
        print(f"Health: {health.status_code}")
        
        # Test projects
        response = requests.get('http://localhost:8000/projects', timeout=5)
        print(f"Projects Status: {response.status_code}")
        
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
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_final()

