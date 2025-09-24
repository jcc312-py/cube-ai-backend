#!/usr/bin/env python3
"""
Quick test to verify the projects API is working
"""

import requests
import json

def test_projects_api():
    """Test the projects API endpoint"""
    print("🧪 Testing Projects API")
    print("=" * 40)
    
    try:
        # Test the projects endpoint
        response = requests.get('http://localhost:8000/projects')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Count: {data.get('count')}")
            print(f"Projects: {len(data.get('projects', []))}")
            
            for project in data.get('projects', []):
                print(f"  - {project.get('name')} ({project.get('file_count', 0)} files)")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_projects_api()

