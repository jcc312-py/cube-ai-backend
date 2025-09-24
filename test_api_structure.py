#!/usr/bin/env python3
"""
Test the API structure to match frontend expectations
"""

import requests
import json
from pathlib import Path

def test_api_structure():
    """Test that the API returns the correct structure"""
    print("🧪 Testing API Structure for Frontend")
    print("=" * 50)
    
    try:
        response = requests.get('http://localhost:8000/projects')
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Count: {data.get('count')}")
            
            projects = data.get('projects', [])
            print(f"Projects found: {len(projects)}")
            
            for i, project in enumerate(projects):
                print(f"\nProject {i+1}:")
                print(f"  Name: {project.get('name')}")
                print(f"  Path: {project.get('path')}")
                print(f"  File Count: {project.get('file_count')}")
                print(f"  Created: {project.get('created')}")
                print(f"  Modified: {project.get('modified')}")
                print(f"  Source: {project.get('source')}")
                print(f"  Type: {project.get('type')}")
                
                # Check if it has the required fields for frontend
                required_fields = ['name', 'path', 'file_count', 'created', 'modified']
                missing_fields = [field for field in required_fields if field not in project]
                if missing_fields:
                    print(f"  ❌ Missing fields: {missing_fields}")
                else:
                    print(f"  ✅ Has all required fields")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_api_structure()

