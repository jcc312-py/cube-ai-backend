#!/usr/bin/env python3
"""
Test the projects endpoint directly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_projects_endpoint():
    """Test the projects endpoint logic directly"""
    print("🧪 Testing Projects Endpoint Logic")
    print("=" * 50)
    
    # Set up the same paths as in main.py
    BASE_DIR = Path(__file__).resolve().parent
    GENERATED_DIR = BASE_DIR / "generated"
    
    print(f"GENERATED_DIR: {GENERATED_DIR}")
    print(f"GENERATED_DIR exists: {GENERATED_DIR.exists()}")
    
    # Test the project listing logic
    projects_dir = GENERATED_DIR / "projects"
    print(f"\nProjects directory: {projects_dir}")
    print(f"Projects directory exists: {projects_dir.exists()}")
    
    projects = []
    
    if projects_dir.exists():
        for project_path in projects_dir.iterdir():
            if project_path.is_dir():
                print(f"\n📁 Processing project: {project_path.name}")
                
                # Count files
                file_count = len(list(project_path.rglob("*.py")))
                total_files = len(list(project_path.rglob("*")))
                print(f"   Python files: {file_count}")
                print(f"   Total files: {total_files}")
                
                # Get project structure
                structure = {
                    "src_files": [],
                    "test_files": [],
                    "other_files": []
                }
                
                for file_path in project_path.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(project_path)
                        if "test" in file_path.name.lower() or "test" in str(rel_path).lower():
                            structure["test_files"].append(str(rel_path))
                        elif file_path.suffix == ".py":
                            structure["src_files"].append(str(rel_path))
                        else:
                            structure["other_files"].append(str(rel_path))
                
                print(f"   Structure: {structure}")
                
                project_info = {
                    "name": project_path.name,
                    "path": str(project_path),
                    "file_count": file_count,
                    "structure": structure,
                    "created": project_path.stat().st_ctime,
                    "modified": project_path.stat().st_mtime
                }
                
                projects.append(project_info)
                print(f"   ✅ Added to projects list")
    
    result = {
        "success": True,
        "projects": projects,
        "count": len(projects)
    }
    
    print(f"\n📊 Final Result:")
    print(f"Success: {result['success']}")
    print(f"Count: {result['count']}")
    print(f"Projects: {[p['name'] for p in result['projects']]}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_projects_endpoint())
