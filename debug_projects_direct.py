#!/usr/bin/env python3
"""
Debug projects directory directly
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_projects_direct():
    """Debug projects directory directly"""
    print("🔍 Debugging Projects Directory Directly")
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
        project_dirs = list(projects_dir.iterdir())
        print(f"Found {len(project_dirs)} items in projects directory:")
        
        for item in project_dirs:
            print(f"  - {item.name} (is_dir: {item.is_dir()})")
            
            if item.is_dir():
                # Count files
                file_count = len(list(item.rglob("*.py")))
                total_files = len(list(item.rglob("*")))
                print(f"    Python files: {file_count}")
                print(f"    Total files: {total_files}")
                
                # Test the project structure logic
                structure = {
                    "src_files": [],
                    "test_files": [],
                    "other_files": []
                }
                
                for file_path in item.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(item)
                        if "test" in file_path.name.lower() or "test" in str(rel_path).lower():
                            structure["test_files"].append(str(rel_path))
                        elif file_path.suffix == ".py":
                            structure["src_files"].append(str(rel_path))
                        else:
                            structure["other_files"].append(str(rel_path))
                
                print(f"    Structure: {structure}")
    else:
        print("❌ Projects directory does not exist!")

if __name__ == "__main__":
    debug_projects_direct()

