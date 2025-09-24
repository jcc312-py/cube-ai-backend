#!/usr/bin/env python3
"""
Debug script to check project listing logic
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_project_listing():
    """Debug the project listing logic"""
    print("🔍 Debugging Project Listing Logic")
    print("=" * 50)
    
    # Set up the same paths as in main.py
    BASE_DIR = Path(__file__).resolve().parent
    GENERATED_DIR = BASE_DIR / "generated"
    
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"GENERATED_DIR: {GENERATED_DIR}")
    print(f"GENERATED_DIR exists: {GENERATED_DIR.exists()}")
    
    # Check projects directory
    projects_dir = GENERATED_DIR / "projects"
    print(f"\nProjects directory: {projects_dir}")
    print(f"Projects directory exists: {projects_dir.exists()}")
    
    if projects_dir.exists():
        print(f"Projects directory contents:")
        for item in projects_dir.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Test the project listing logic
        print(f"\nTesting project listing logic:")
        all_projects = []
        
        for project_path in projects_dir.iterdir():
            if project_path.is_dir():
                print(f"\n  Processing project: {project_path.name}")
                
                # Count files
                file_count = len(list(project_path.rglob("*.py")))
                print(f"    Python files: {file_count}")
                
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
                
                print(f"    Structure: {structure}")
                
                project_info = {
                    "name": project_path.name,
                    "description": f"Git repository: {project_path.name}",
                    "path": str(project_path),
                    "file_count": file_count,
                    "structure": structure,
                    "created_at": project_path.stat().st_ctime,
                    "status": "active",
                    "source": "git_pull",
                    "type": "git_repository",
                    "github_repo": None
                }
                
                all_projects.append(project_info)
                print(f"    Added to projects list")
        
        print(f"\nTotal projects found: {len(all_projects)}")
        for project in all_projects:
            print(f"  - {project['name']}: {project['file_count']} files")
    
    else:
        print("❌ Projects directory does not exist!")

if __name__ == "__main__":
    debug_project_listing()
