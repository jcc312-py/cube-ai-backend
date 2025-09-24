#!/usr/bin/env python3
"""
Simple test to verify project file access works
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_project_file_access():
    """Test that we can access files from the projects directory"""
    print("🧪 Testing Project File Access (Local)")
    print("=" * 50)
    
    # Check if projects directory exists
    projects_dir = Path("generated/projects")
    print(f"Projects directory: {projects_dir}")
    print(f"Exists: {projects_dir.exists()}")
    
    if not projects_dir.exists():
        print("❌ Projects directory not found!")
        return
    
    # List all projects
    projects = []
    for project_path in projects_dir.iterdir():
        if project_path.is_dir():
            print(f"\n📁 Project: {project_path.name}")
            print(f"   Path: {project_path}")
            
            # Count files
            all_files = list(project_path.rglob("*"))
            files = [f for f in all_files if f.is_file()]
            print(f"   Total files: {len(files)}")
            
            # Show files
            for file_path in files[:5]:  # Show first 5 files
                rel_path = file_path.relative_to(project_path)
                size = file_path.stat().st_size
                print(f"   📄 {rel_path} ({size} bytes)")
            
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
            
            projects.append({
                "name": project_path.name,
                "path": str(project_path),
                "file_count": len(files)
            })
    
    print(f"\n✅ Found {len(projects)} projects")
    
    # Test reading a specific file
    if projects:
        test_project = projects[0]
        project_path = Path(test_project["path"])
        print(f"\n🔍 Testing file reading from: {test_project['name']}")
        
        # Find a Python file
        py_files = list(project_path.rglob("*.py"))
        if py_files:
            test_file = py_files[0]
            print(f"   Reading: {test_file.name}")
            try:
                content = test_file.read_text(encoding='utf-8')
                lines = len(content.splitlines())
                print(f"   ✅ Successfully read {len(content)} characters, {lines} lines")
                print(f"   Preview: {content[:200]}...")
            except Exception as e:
                print(f"   ❌ Failed to read file: {e}")
        else:
            print("   ⚠️ No Python files found")
    
    print("\n🎉 Local file access test completed!")

if __name__ == "__main__":
    test_project_file_access()
