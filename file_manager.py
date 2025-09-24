"""
Advanced File Management System for AI-Generated Code
Handles automatic file saving, GitHub integration, and project organization
"""

import os
import re
import json
import uuid
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Import Git integration if available
try:
    import sys
    from pathlib import Path
    git_integration_path = Path(__file__).parent.parent / "git-integration"
    sys.path.append(str(git_integration_path))
    from github_service import GitHubService
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    GitHubService = None

class FileManager:
    """Advanced file management for AI-generated code with automatic Git integration"""
    
    def __init__(self, base_dir: str = "generated"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize GitHub service if available
        self.github_service = None
        if GIT_AVAILABLE:
            try:
                self.github_service = GitHubService()
                logging.info("✅ GitHub service initialized")
            except Exception as e:
                logging.warning(f"⚠️ GitHub service not available: {e}")
        
        # Project tracking
        self.active_projects = {}
        
    def create_project(self, task_description: str, conversation_id: str = None) -> Dict[str, str]:
        """Create a new project with organized folder structure"""
        try:
            # Generate project name from task description
            project_name = self._generate_project_name(task_description)
            project_id = str(uuid.uuid4())[:8]
            full_project_name = f"{project_name}_{project_id}"
            
            # Create project directory structure
            project_dir = self.base_dir / "projects" / full_project_name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (project_dir / "src").mkdir(exist_ok=True)
            (project_dir / "tests").mkdir(exist_ok=True)
            (project_dir / "docs").mkdir(exist_ok=True)
            
            # Create project metadata
            metadata = {
                "project_id": project_id,
                "project_name": full_project_name,
                "task_description": task_description,
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "files": {
                    "src": [],
                    "tests": [],
                    "docs": []
                },
                "github_repo": None,
                "status": "active"
            }
            
            # Save metadata
            with open(project_dir / "project.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Track active project
            self.active_projects[conversation_id or "default"] = {
                "project_dir": project_dir,
                "metadata": metadata
            }
            
            logging.info(f"📁 Created project: {full_project_name}")
            return {
                "project_name": full_project_name,
                "project_dir": str(project_dir),
                "project_id": project_id
            }
            
        except Exception as e:
            logging.error(f"❌ Failed to create project: {e}")
            return {}
    
    def save_code(self, code: str, filename: str = None, file_type: str = "src", 
                  conversation_id: str = None, task_description: str = None) -> Dict[str, str]:
        """Save generated code with automatic project management and enhanced error handling"""
        try:
            # Validate inputs
            if not code or not code.strip():
                return {"error": "Code content cannot be empty"}
            
            if file_type not in ["src", "tests", "docs"]:
                return {"error": f"Invalid file_type '{file_type}'. Must be 'src', 'tests', or 'docs'"}
            
            # Validate code syntax for Python files
            if filename and filename.endswith('.py'):
                syntax_valid, syntax_error = self._validate_python_syntax(code)
                if not syntax_valid:
                    logging.warning(f"⚠️ Python syntax warning: {syntax_error}")
                    # Continue saving but log the warning
            
            # Get or create project
            if conversation_id and conversation_id in self.active_projects:
                project_info = self.active_projects[conversation_id]
                project_dir = project_info["project_dir"]
                metadata = project_info["metadata"]
            else:
                # Create new project
                project_info = self.create_project(task_description or "AI Generated Code", conversation_id)
                if not project_info:
                    return {"error": "Failed to create project"}
                project_dir = Path(project_info["project_dir"])
                metadata = self._load_project_metadata(project_dir)
            
            # Generate filename if not provided
            if not filename:
                filename = self._generate_filename(code, file_type)
            
            # Validate filename
            if not self._is_valid_filename(filename):
                return {"error": f"Invalid filename: {filename}"}
            
            # Determine file type and directory
            file_dir = project_dir / file_type
            filepath = file_dir / filename
            
            # Ensure directory exists
            file_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if file already exists and create backup
            if filepath.exists():
                backup_path = filepath.with_suffix(f"{filepath.suffix}.backup_{int(datetime.now().timestamp())}")
                filepath.rename(backup_path)
                logging.info(f"📋 Created backup: {backup_path}")
            
            # Save code to file with atomic write
            temp_path = filepath.with_suffix(f"{filepath.suffix}.tmp")
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                # Atomic move
                temp_path.rename(filepath)
                
            except Exception as write_error:
                # Clean up temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                raise write_error
            
            # Update metadata
            file_info = {
                "filename": filename,
                "filepath": str(filepath),
                "created_at": datetime.now().isoformat(),
                "size": len(code),
                "lines": len(code.splitlines()),
                "encoding": "utf-8"
            }
            
            # Add syntax validation info for Python files
            if filename.endswith('.py'):
                file_info["syntax_valid"] = syntax_valid
                if not syntax_valid:
                    file_info["syntax_error"] = syntax_error
            
            metadata["files"][file_type].append(file_info)
            
            # Save updated metadata with atomic write
            metadata_temp = project_dir / "project.json.tmp"
            with open(metadata_temp, 'w') as f:
                json.dump(metadata, f, indent=2)
            metadata_temp.rename(project_dir / "project.json")
            
            logging.info(f"💾 Code saved to: {filepath} ({len(code)} chars, {file_info['lines']} lines)")
            
            # Auto-upload to GitHub if available (async, don't block)
            github_result = self._auto_upload_to_github(project_dir, metadata)
            
            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "project_name": metadata["project_name"],
                "file_info": file_info,
                "github_result": github_result
            }
            
        except PermissionError as e:
            logging.error(f"❌ Permission denied saving file: {e}")
            return {"error": f"Permission denied: {str(e)}"}
        except OSError as e:
            logging.error(f"❌ File system error saving file: {e}")
            return {"error": f"File system error: {str(e)}"}
        except Exception as e:
            logging.error(f"❌ Failed to save code: {e}")
            return {"error": str(e)}
    
    def _generate_project_name(self, task_description: str) -> str:
        """Generate a sensible project name from task description"""
        # Clean and normalize the description
        clean_desc = re.sub(r'[^\w\s-]', '', task_description.lower())
        clean_desc = re.sub(r'\s+', '_', clean_desc.strip())
        
        # Limit length and add meaningful suffix
        if len(clean_desc) > 30:
            clean_desc = clean_desc[:30]
        
        # Add descriptive suffix based on content
        if any(word in clean_desc for word in ['sum', 'add', 'calculate', 'math']):
            clean_desc += "_calculator"
        elif any(word in clean_desc for word in ['app', 'application', 'program']):
            clean_desc += "_app"
        elif any(word in clean_desc for word in ['api', 'service', 'server']):
            clean_desc += "_api"
        else:
            clean_desc += "_project"
        
        return clean_desc
    
    def _generate_filename(self, code: str, file_type: str) -> str:
        """Generate a sensible filename based on code content"""
        # Extract function/class names
        function_match = re.search(r'def\s+(\w+)', code)
        class_match = re.search(r'class\s+(\w+)', code)
        
        if function_match:
            base_name = function_match.group(1)
        elif class_match:
            base_name = class_match.group(1)
        else:
            base_name = "main"
        
        # Add appropriate extension
        if file_type == "tests":
            return f"test_{base_name}.py"
        elif file_type == "docs":
            return f"{base_name}_documentation.md"
        else:
            return f"{base_name}.py"
    
    def _auto_upload_to_github(self, project_dir: Path, metadata: Dict) -> Dict[str, str]:
        """Automatically upload project to GitHub with retry logic and better error handling"""
        if not self.github_service:
            return {"status": "github_not_available", "message": "GitHub service not initialized"}
        
        try:
            # Check if repo already exists
            if metadata.get("github_repo"):
                return {"status": "already_uploaded", "repo": metadata["github_repo"]}
            
            # Create GitHub repository with retry logic
            repo_name = metadata["project_name"]
            repo_description = f"AI Generated: {metadata['task_description'][:100]}"
            
            # Sanitize repo name for GitHub
            repo_name = self._sanitize_repo_name(repo_name)
            
            # Create repository with retry
            repo_result = self._create_repository_with_retry(repo_name, repo_description)
            
            if not repo_result.get("success"):
                return {"status": "failed", "error": repo_result.get("error")}
            
            # Upload all files with retry logic
            upload_result = self._upload_project_files_with_retry(project_dir, repo_name)
            
            if upload_result.get("success"):
                # Update metadata
                metadata["github_repo"] = repo_result["repo_url"]
                metadata["status"] = "uploaded"
                metadata["last_upload"] = datetime.now().isoformat()
                
                # Save metadata atomically
                metadata_temp = project_dir / "project.json.tmp"
                with open(metadata_temp, 'w') as f:
                    json.dump(metadata, f, indent=2)
                metadata_temp.rename(project_dir / "project.json")
                
                logging.info(f"🐙 Project uploaded to GitHub: {repo_result['repo_url']}")
                return {
                    "status": "success",
                    "repo_url": repo_result["repo_url"],
                    "files_uploaded": upload_result.get("files_uploaded", 0),
                    "upload_time": metadata["last_upload"]
                }
            else:
                return {"status": "upload_failed", "error": upload_result.get("error")}
                
        except Exception as e:
            logging.error(f"❌ GitHub upload failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _upload_project_files(self, project_dir: Path, repo_name: str) -> Dict:
        """Upload all project files to GitHub"""
        try:
            files_uploaded = 0
            
            # Upload files from each directory
            for subdir in ["src", "tests", "docs"]:
                subdir_path = project_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.iterdir():
                        if file_path.is_file():
                            # Read file content
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Upload to GitHub
                            github_path = f"{subdir}/{file_path.name}"
                            upload_result = self.github_service.push_file(
                                repo_name=repo_name,
                                file_path=github_path,
                                content=content,
                                commit_message=f"Add {file_path.name}"
                            )
                            
                            if upload_result.get("success"):
                                files_uploaded += 1
                            else:
                                logging.warning(f"Failed to upload {file_path.name}: {upload_result.get('error')}")
            
            # Upload project metadata
            metadata_path = project_dir / "project.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                upload_result = self.github_service.push_file(
                    repo_name=repo_name,
                    file_path="project.json",
                    content=content,
                    commit_message="Add project metadata"
                )
                
                if upload_result.get("success"):
                    files_uploaded += 1
            
            return {
                "success": True,
                "files_uploaded": files_uploaded
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_project_metadata(self, project_dir: Path) -> Dict:
        """Load project metadata from file"""
        metadata_path = project_dir / "project.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _validate_python_syntax(self, code: str) -> Tuple[bool, str]:
        """Validate Python syntax and return (is_valid, error_message)"""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename is valid for the filesystem"""
        if not filename or not filename.strip():
            return False
        
        # Check for invalid characters
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            return False
        
        # Check for reserved names on Windows
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        if filename.upper().split('.')[0] in reserved_names:
            return False
        
        # Check length
        if len(filename) > 255:
            return False
        
        return True
    
    def _sanitize_repo_name(self, name: str) -> str:
        """Sanitize repository name for GitHub"""
        # Remove invalid characters and replace with hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '-', name)
        # Remove consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        # Ensure it starts with a letter or number
        if not re.match(r'^[a-zA-Z0-9]', sanitized):
            sanitized = 'repo-' + sanitized
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        return sanitized
    
    def _create_repository_with_retry(self, repo_name: str, description: str, max_retries: int = 3) -> Dict:
        """Create GitHub repository with retry logic"""
        for attempt in range(max_retries):
            try:
                result = self.github_service.create_repository(
                    name=repo_name,
                    description=description,
                    private=False
                )
                
                if result.get("success"):
                    return result
                
                # If repo already exists, try with a different name
                if "already exists" in str(result.get("error", "")).lower():
                    repo_name = f"{repo_name}-{int(datetime.now().timestamp())}"
                    continue
                
                # For other errors, wait and retry
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def _upload_project_files_with_retry(self, project_dir: Path, repo_name: str, max_retries: int = 3) -> Dict:
        """Upload project files with retry logic"""
        for attempt in range(max_retries):
            try:
                result = self._upload_project_files(project_dir, repo_name)
                if result.get("success"):
                    return result
                
                # Wait and retry on failure
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def get_project_files(self, project_name: str) -> List[Dict]:
        """Get all files in a project"""
        project_dir = self.base_dir / "projects" / project_name
        if not project_dir.exists():
            return []
        
        files = []
        for subdir in ["src", "tests", "docs"]:
            subdir_path = project_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.iterdir():
                    if file_path.is_file():
                        files.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "type": subdir,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
        
        return files
    
    def list_projects(self) -> List[Dict]:
        """List all projects"""
        projects = []
        projects_dir = self.base_dir / "projects"
        
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    metadata = self._load_project_metadata(project_dir)
                    if metadata:
                        projects.append({
                            "name": metadata["project_name"],
                            "description": metadata["task_description"],
                            "created_at": metadata["created_at"],
                            "status": metadata["status"],
                            "github_repo": metadata.get("github_repo"),
                            "file_count": sum(len(files) for files in metadata["files"].values())
                        })
        
        return projects

# Global file manager instance
file_manager = FileManager()

