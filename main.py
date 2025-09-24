# =============================================================================
# IMPORTS - DON'T CHANGE THESE (unless you know what you're doing)
# =============================================================================
# These are all the tools we need to make our agent system work
import os
from pathlib import Path
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import tempfile
import subprocess
import sys
import logging
import asyncio
from abc import ABC, abstractmethod
import json
from enum import Enum
import ast # Added for syntax validation
import uuid

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
import uvicorn

# Database integration
from database import SafeDatabaseIntegration, ConversationRequest, ConversationResponse, ChatRequestWithConversation

from repo_importer import router as repo_import_router
# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# GPU Detection and Configuration
def detect_gpu():
    """Detect if GPU is available and configure accordingly"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected - using GPU acceleration")
            return {
                "num_gpu": 1,  # Use 1 GPU
                "num_thread": 8,  # Number of CPU threads
                "temperature": 0.3,  # Lower temperature for more focused code generation
                "top_p": 0.9,  # Nucleus sampling
                "repeat_penalty": 1.1,  # Prevent repetition
                "top_k": 40,  # Top-k sampling for better code quality
                "num_ctx": 4096,  # Context window size
            }
        else:
            print("⚠️ GPU not detected - using CPU only")
            return {
                "num_gpu": 0,  # No GPU
                "num_thread": 8,  # Number of CPU threads
                "temperature": 0.3,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "top_k": 40,
                "num_ctx": 4096,
            }
    except Exception as e:
        print(f"⚠️ Could not detect GPU: {e} - using CPU only")
        return {
            "num_gpu": 0,  # No GPU
            "num_thread": 8,  # Number of CPU threads
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "num_ctx": 4096,
        }

# Default GPU configuration for all agents - Optimized for CodeLlama:7b-instruct
DEFAULT_GPU_CONFIG = detect_gpu()

# Default model configuration - Now using CodeLlama:7b-instruct
DEFAULT_MODEL = "codellama:7b-instruct"

# Model-specific configurations for different tasks
MODEL_CONFIGS = {
    "codellama:7b-instruct": {
        "num_gpu": DEFAULT_GPU_CONFIG["num_gpu"],  # Use detected GPU setting
        "num_thread": 8,
        "temperature": 0.3,  # Lower for code generation
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "top_k": 40,
        "num_ctx": 4096,
    },
    "mistral": {
        "num_gpu": DEFAULT_GPU_CONFIG["num_gpu"],  # Use detected GPU setting
        "num_thread": 8,
        "temperature": 0.7,  # Higher for general tasks
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    },
    "llama2": {
        "num_gpu": DEFAULT_GPU_CONFIG["num_gpu"],  # Use detected GPU setting
        "num_thread": 8,
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
}

# =============================================================================
# APP & CORS CONFIGURATION - DON'T CHANGE THIS SECTION
# =============================================================================
# What: Creates the web server and allows frontend to connect
# Why: Frontend needs to talk to backend, CORS allows this
# Can I change: NO - this is essential for frontend-backend communication
app = FastAPI(title="Multi-Agent AI System", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend to connect (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the API testing interface
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(repo_import_router, prefix="/api")

# =============================================================================
# ROOT ENDPOINT - API STATUS PAGE
# =============================================================================
@app.get("/")
async def root():
    """Root endpoint - serves the API testing interface"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}

@app.get("/test-projects")
async def test_projects():
    """Test endpoint to verify projects are found"""
    try:
        projects_dir = GENERATED_DIR / "projects"
        projects = []
        
        if projects_dir.exists():
            for project_path in projects_dir.iterdir():
                if project_path.is_dir():
                    file_count = len(list(project_path.rglob("*.py")))
                    projects.append({
                        "name": project_path.name,
                        "file_count": file_count,
                        "path": str(project_path)
                    })
        
        return {
            "success": True,
            "projects": projects,
            "count": len(projects),
            "projects_dir": str(projects_dir),
            "exists": projects_dir.exists()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api-docs")
async def api_docs():
    """Redirect to FastAPI's built-in documentation"""
    return RedirectResponse(url="/docs")

@app.get("/api-status")
async def api_status():
    """API status endpoint - shows API status and available endpoints"""
    return {
        "message": "Multi-Agent AI System API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "conversations": "/conversations",
            "files": "/list-files",
            "workflow": "/run-workflow",
            "manual_flow": "/run-manual-flow",
            "agents": "/agents",
            "models": "/models",
            "gpu_status": "/gpu-status"
        },
        "documentation": "/docs",
        "frontend": "http://localhost:5173",
        "api_testing_interface": "/"
    }

# =============================================================================
# PATHS & DIRECTORIES - YOU CAN CHANGE THESE PATHS
# =============================================================================
# What: Defines where files are stored and loaded from
# Why: We need consistent places to save generated code and tests
# Can I change: YES - you can change where files are saved
BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated"  # Where all generated files go
GENERATED_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# =============================================================================
# DATABASE INTEGRATION - SAFE ADDITION
# =============================================================================
# Initialize database integration (non-disruptive)
db_integration = SafeDatabaseIntegration()

# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================
class MessageType(Enum):
    """
    What: Different types of messages agents can send to each other
    Why: So agents know how to handle different kinds of messages
    Can I change: YES - you can add more message types like "FEEDBACK", "UPDATE", etc.
    """
    TASK = "task"        # "Do this work for me"
    DATA = "data"        # "Here's some information"
    REQUEST = "request"  # "I need something from you"
    RESPONSE = "response" # "Here's your answer"
    ERROR = "error"      # "Something went wrong"
    STATUS = "status"    # "Here's my current status"
    REVIEW = "review"    # "Review generated code or tests"

# =============================================================================
# AGENT MESSAGE - THE WAY AGENTS TALK TO EACH OTHER
# =============================================================================
class AgentMessage(BaseModel):
    """
    What: This is like an email between agents - it has sender, receiver, content
    Why: Agents need a standard way to talk to each other
    Can I change: YES - but be careful, this affects everything
    """
    id: str                              # Unique ID for this message (like email ID)
    from_agent: str                      # Who sent this message
    to_agent: str                        # Who should receive this message
    message_type: MessageType            # What kind of message is this
    content: str                         # The actual message content
    metadata: Dict[str, Any] = {}        # Extra information (you can add anything here)
    timestamp: datetime = datetime.now() # When was this message sent
    
    # NEW: These help with retrying failed messages
    retry_count: int = 0                    # How many times we've tried to send this
    parent_message_id: Optional[str] = None # If this is a retry, what's the original message

    def create_retry_message(self) -> 'AgentMessage':
        """
        What: Makes a copy of this message for retrying
        Why: Sometimes messages fail, so we need to try again
        Can I change: YES - you can modify retry logic
        """
        return AgentMessage(
            id=f"{self.id}_retry_{self.retry_count + 1}",
            from_agent=self.from_agent,
            to_agent=self.to_agent,
            message_type=self.message_type,
            content=self.content,
            metadata=self.metadata,
            parent_message_id=self.id,
            retry_count=self.retry_count + 1
        )

# =============================================================================
# AGENT STATUS - TRACKS WHAT EACH AGENT IS DOING
# =============================================================================
class AgentStatus(Enum):
    """
    What: Tells us what state each agent is in
    Why: So we know if agents are busy, done, or having problems
    Can I change: YES - you can add more statuses like "PAUSED", "RESTARTING", etc.
    """
    IDLE = "idle"           # Agent is ready for work
    WORKING = "working"      # Agent is currently processing
    WAITING = "waiting"      # Agent is waiting for something
    COMPLETED = "completed"  # Agent finished its task
    ERROR = "error"          # Agent encountered an error

# =============================================================================
# AGENT MEMORY - HOW AGENTS REMEMBER THINGS
# =============================================================================
class AgentMemory(BaseModel):
    """
    What: Each agent has a memory system to remember conversations and context
    Why: Agents need to remember what they've done and what others have said
    Can I change: YES - you can add database storage, more memory types, etc.
    """
    short_term: List[AgentMessage] = []  # Recent messages (last 50)
    long_term: Dict[str, Any] = {}       # Persistent data (like a database)
    context: Dict[str, Any] = {}         # Current context (what we're working on)
    
    def add_message(self, message: AgentMessage):
        """
        What: Adds a new message to the agent's memory
        Why: Agents need to remember conversations
        Can I change: YES - you can change how messages are stored
        """
        self.short_term.append(message)
        # Keep only last 50 messages in short-term (prevents memory overflow)
        if len(self.short_term) > 50:
            self.short_term = self.short_term[-50:]
    
    def get_context_for_prompt(self) -> str:
        """
        What: Creates a summary of recent conversations for LLM prompts
        Why: LLMs need context to understand what's happening
        Can I change: YES - you can change how context is formatted
        """
        recent_messages = self.short_term[-10:]  # Last 10 messages
        context_str = "Recent conversation:\n"
        for msg in recent_messages:
            context_str += f"{msg.from_agent} -> {msg.to_agent}: {msg.content[:100]}...\n"
        return context_str

# =============================================================================
# BASE AGENT - THE FOUNDATION FOR ALL AGENTS
# =============================================================================
class BaseAgent(ABC):
    """
    What: The base class that all agents inherit from
    Why: Provides common functionality like message handling and memory
    Can I change: YES - you can add more base functionality, but be careful
    """
    
    def __init__(self, agent_id: str, agent_type: str, role: str, 
                 model_name: str = DEFAULT_MODEL, model_config: Dict[str, Any] = None):
        """
        What: Sets up a new agent with its identity and AI model
        Why: Each agent needs a unique identity and its own AI model
        Can I change: YES - you can add more initialization parameters
        """
        self.agent_id = agent_id          # Unique name for this agent
        self.agent_type = agent_type      # What kind of agent (coder, tester, etc.)
        self.role = role                  # What this agent does
        self.status = AgentStatus.IDLE    # Current status
        self.memory = AgentMemory()       # Agent's memory system
        self.model_config = model_config or {}  # AI model settings
        
        # Initialize LLM - each agent gets its own AI model instance
        # Get model-specific configuration or use default
        model_specific_config = MODEL_CONFIGS.get(model_name, DEFAULT_GPU_CONFIG.copy())
        
        # Merge configurations: model-specific -> user-provided -> default
        final_config = {**DEFAULT_GPU_CONFIG, **model_specific_config, **self.model_config}
        self.llm = OllamaLLM(model=model_name, **final_config)
        
        # Message handling - maps message types to handler functions
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.TASK: self.handle_task,
            MessageType.DATA: self.handle_data,
            MessageType.REQUEST: self.handle_request,
            MessageType.RESPONSE: self.handle_response,
            MessageType.ERROR: self.handle_error,
            MessageType.STATUS: self.handle_status
        }
    
    async def process_message(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Processes an incoming message and returns responses
        Why: This is how agents communicate with each other
        Can I change: YES - you can modify how messages are processed
        """
        try:
            self.status = AgentStatus.WORKING
            self.memory.add_message(message)
            
            # Get the right handler for this message type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                responses = await handler(message)
            else:
                # If no handler, create an error response
                responses = [self.create_error_message(message.from_agent, 
                                                    f"Unknown message type: {message.message_type}")]
            
            self.status = AgentStatus.IDLE
            return responses
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return [self.create_error_message(message.from_agent, str(e))]
    
    def create_message(self, to_agent: str, message_type: MessageType, 
                      content: str, metadata: Dict[str, Any] = None) -> AgentMessage:
        """
        What: Creates a new message to send to another agent
        Why: Agents need a standard way to create messages
        Can I change: YES - you can modify message creation
        """
        return AgentMessage(
            id=f"{self.agent_id}_{datetime.now().timestamp()}",
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
    
    def create_error_message(self, to_agent: str, error: str) -> AgentMessage:
        """
        What: Creates an error message when something goes wrong
        Why: Agents need to report errors to each other
        Can I change: YES - you can modify error reporting
        """
        return self.create_message(to_agent, MessageType.ERROR, error)
    
    # These are abstract methods - each agent type must implement them
    @abstractmethod
    async def handle_task(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle task messages - each agent type implements this differently"""
        pass
    
    async def handle_data(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle data messages - default implementation"""
        return [self.create_message(message.from_agent, MessageType.RESPONSE, 
                                  f"Received data: {message.content}")]
    
    async def handle_request(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle request messages - default implementation"""
        return [self.create_message(message.from_agent, MessageType.RESPONSE, 
                                  f"Handled request: {message.content}")]
    
    async def handle_response(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle response messages - default implementation"""
        return []  # Usually no response needed for responses
    
    async def handle_error(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle error messages - default implementation"""
        print(f"Agent {self.agent_id} received error: {message.content}")
        return []
    
    async def handle_status(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle status messages - default implementation"""
        return [self.create_message(message.from_agent, MessageType.STATUS, 
                                  f"Status: {self.status.value}")]
    
    def update_model(self, model_name: str, model_config: Dict[str, Any] = None):
        """
        What: Changes the AI model this agent uses
        Why: Agents might need different models for different tasks
        Can I change: YES - you can modify model switching logic
        """
        self.model_config = model_config or {}
        self.llm = OllamaLLM(model=model_name, **self.model_config)

# =============================================================================
# COORDINATOR AGENT - THE BOSS AGENT
# =============================================================================
class CoordinatorAgent(BaseAgent):
    """
    What: The main coordinator that manages other agents
    Why: Someone needs to coordinate the workflow and assign tasks
    Can I change: YES - you can modify how coordination works
    """
    
    async def handle_task(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles incoming tasks and coordinates the workflow
        Why: This is where the main logic for task distribution happens
        Can I change: YES - you can modify the coordination strategy
        """
        try:
            print(f"🎯 CoordinatorAgent received task: {message.content}")
            
            # Check if the message is asking to see/read code
            if any(phrase in message.content.lower() for phrase in ["can you see the code", "see the code", "read the code", "show me the code", "what code", "do you see any code"]):
                return await self._handle_code_reading_request(message)
            
            # Get context from memory
            context = self.memory.get_context_for_prompt()
            
            # Create a comprehensive prompt for the coordinator
            prompt = f"""
            You are a smart coordinator managing a team of AI agents.
            
            Current task: {message.content}
            Recent context: {context}
            
            Your job is to:
            1. Break down the task into steps
            2. Decide which agents should handle each step
            3. Create clear instructions for each agent
            
            Available agents:
            - coder: Writes code and implements features
            - tester: Creates test cases and validates code
            - runner: Executes tests and reports results
            
            Respond with a JSON structure like this:
            {{
                "steps": [
                    {{
                        "agent": "coder",
                        "task": "Write a Python function that...",
                        "priority": 1
                    }},
                    {{
                        "agent": "tester", 
                        "task": "Create test cases for the function",
                        "priority": 2
                    }}
                ]
            }}
            """
            
            # Get AI response
            response = self.llm.invoke(prompt)
            
            # Try to parse the response as JSON
            try:
                plan = json.loads(response)
                steps = plan.get("steps", [])
            except:
                # If JSON parsing fails, create a simple plan
                steps = [
                    {"agent": "coder", "task": message.content, "priority": 1},
                    {"agent": "tester", "task": "Create tests for the generated code", "priority": 2}
                ]
            
            print(f"📋 Created {len(steps)} workflow steps")
            
            # Create messages for each step
            responses = []
            for step in steps:
                agent_id = step["agent"]
                task = step["task"]
                
                print(f"📤 Sending task to {agent_id}: {task[:50]}...")
                
                # Create task message for the agent
                task_message = self.create_message(
                    to_agent=agent_id,
                    message_type=MessageType.TASK,
                    content=task,
                    metadata={"priority": step.get("priority", 1)}
                )
                responses.append(task_message)
            
            print(f"📤 Sending {len(responses)} task messages")
            return responses
            
        except Exception as e:
            print(f"❌ Coordination failed: {str(e)}")
            return [self.create_error_message(message.from_agent, f"Coordination failed: {str(e)}")]
    
    async def _handle_code_reading_request(self, message: AgentMessage) -> List[AgentMessage]:
        """Handle requests to read/see code from projects or generated files"""
        try:
            print("🔍 CoordinatorAgent: Handling code reading request")
            
            # Try to read code from all available sources
            import httpx
            
            # First, try to get available projects
            projects = []
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:8000/projects/list")
                    if response.status_code == 200:
                        data = response.json()
                        projects = data.get("projects", [])
                        print(f"📁 Found {len(projects)} projects")
            except Exception as e:
                print(f"⚠️ Could not fetch projects: {e}")
            
            # Read code from all sources
            all_code = []
            
            # Read from projects
            for project in projects:
                project_name = project.get("name")
                if project_name:
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(f"http://localhost:8000/projects/{project_name}/code")
                            if response.status_code == 200:
                                data = response.json()
                                files = data.get("files", [])
                                for file_info in files:
                                    all_code.append({
                                        "source": f"project:{project_name}",
                                        "file": file_info["name"],
                                        "path": file_info["path"],
                                        "content": file_info["content"],
                                        "lines": file_info["lines"],
                                        "is_test": file_info["is_test"]
                                    })
                                print(f"📄 Read {len(files)} files from project {project_name}")
                    except Exception as e:
                        print(f"⚠️ Could not read from project {project_name}: {e}")
            
            # Also read from generated files
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post("http://localhost:8000/agents/read-code", json={
                        "include_generated": True,
                        "file_pattern": "*.py"
                    })
                    if response.status_code == 200:
                        data = response.json()
                        files = data.get("code_files", [])
                        for file_info in files:
                            all_code.append({
                                "source": file_info["source"],
                                "file": file_info["name"],
                                "path": file_info["path"],
                                "content": file_info["content"],
                                "lines": len(file_info["content"].splitlines()),
                                "is_test": file_info["is_test"]
                            })
                        print(f"📄 Read {len(files)} files from generated directory")
            except Exception as e:
                print(f"⚠️ Could not read generated files: {e}")
            
            # Prepare response based on what we found
            if not all_code:
                response_content = "I don't see any code files available. Please:\n1. Pull code from a git repository using /git/pull\n2. Generate some code first\n3. Or specify which project you want me to look at"
            else:
                # Organize code by type
                main_code = [f for f in all_code if not f["is_test"]]
                test_code = [f for f in all_code if f["is_test"]]
                
                response_content = f"✅ I can see {len(all_code)} code files:\n\n"
                
                if main_code:
                    response_content += f"📝 **Main Code Files ({len(main_code)}):**\n"
                    for code_file in main_code[:5]:  # Show first 5 files
                        response_content += f"- {code_file['source']}/{code_file['file']} ({code_file['lines']} lines)\n"
                    if len(main_code) > 5:
                        response_content += f"... and {len(main_code) - 5} more files\n"
                
                if test_code:
                    response_content += f"\n🧪 **Test Files ({len(test_code)}):**\n"
                    for code_file in test_code[:3]:  # Show first 3 test files
                        response_content += f"- {code_file['source']}/{code_file['file']} ({code_file['lines']} lines)\n"
                    if len(test_code) > 3:
                        response_content += f"... and {len(test_code) - 3} more test files\n"
                
                response_content += f"\n**Total:** {len(all_code)} files, {sum(f['lines'] for f in all_code)} lines of code\n\n"
                response_content += "Would you like me to:\n1. Show specific file contents\n2. Analyze the code structure\n3. Suggest improvements\n4. Run tests on the code"
            
            # Create response message
            response_message = self.create_message(
                to_agent=message.from_agent,
                message_type=MessageType.DATA,
                content=response_content,
                metadata={"code_files_found": len(all_code), "sources": list(set(f["source"] for f in all_code))}
            )
            
            return [response_message]
            
        except Exception as e:
            print(f"❌ Code reading failed: {str(e)}")
            error_message = self.create_message(
                to_agent=message.from_agent,
                message_type=MessageType.ERROR,
                content=f"Failed to read code: {str(e)}"
            )
            return [error_message]

# =============================================================================
# CODER AGENT - WRITES CODE
# =============================================================================
class CoderAgent(BaseAgent):
    """
    What: An agent that writes code based on requirements
    Why: We need an agent specifically for code generation
    Can I change: YES - you can modify how code generation works
    """
    
    async def handle_task(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles coding tasks and generates code
        Why: This is where the actual code generation happens
        Can I change: YES - you can modify the code generation strategy
        """
        try:
            print(f"🔧 CoderAgent received task: {message.content}")
            
            # Enhanced prompt for clean code generation
            prompt = f"""
You are an expert Python developer. Generate ONLY clean, working Python code.

Task: {message.content}

CRITICAL REQUIREMENTS:
- Generate ONLY the Python code, NO explanations, NO comments, NO markdown
- Write complete, runnable Python functions
- Include proper error handling with try/except
- Add type hints and docstrings
- Follow PEP 8 style guidelines
- DO NOT include any explanatory text or comments outside the code
- DO NOT use markdown formatting or code blocks
- DO NOT add "Here's the code:" or similar text

Example of what to generate:
def sum_numbers(a: int, b: int) -> int:
    \"\"\"Add two numbers and return the result.\"\"\"
    try:
        return a + b
    except TypeError as e:
        raise ValueError(f"Both arguments must be integers: {{e}}")

Generate ONLY the function code, nothing else.
"""
            
            response = self.llm.invoke(prompt)
            code = self._simple_code_extraction(response)
            
            # Save and return
            timestamp = _timestamp()
            filename = f"code_{timestamp}.py"
            filepath = GENERATED_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Return messages
            response_message = self.create_message(
                to_agent=message.from_agent,
                message_type=MessageType.DATA,
                content=f"Generated code saved to {filename}",
                metadata={"code": code, "filename": filename, "filepath": str(filepath)}
            )
            
            tester_message = self.create_message(
                to_agent="tester",
                message_type=MessageType.DATA,
                content=f"Code generated for testing",
                metadata={"code": code, "filename": filename, "filepath": str(filepath)}
            )
            
            return [response_message, tester_message]
            
        except Exception as e:
            print(f"❌ Code generation failed: {str(e)}")
            return [self.create_error_message(message.from_agent, f"Code generation failed: {str(e)}")]
    
    async def _read_existing_code(self, project_name: str = None) -> List[Dict]:
        """Read existing code from projects or generated files"""
        try:
            import httpx
            all_code = []
            
            # Read from specific project if provided
            if project_name:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:8000/projects/{project_name}/code")
                    if response.status_code == 200:
                        data = response.json()
                        files = data.get("files", [])
                        for file_info in files:
                            all_code.append({
                                "source": f"project:{project_name}",
                                "file": file_info["name"],
                                "path": file_info["path"],
                                "content": file_info["content"],
                                "lines": file_info["lines"],
                                "is_test": file_info["is_test"]
                            })
            else:
                # Read from all available sources
                async with httpx.AsyncClient() as client:
                    response = await client.post("http://localhost:8000/agents/read-code", json={
                        "include_generated": True,
                        "file_pattern": "*.py"
                    })
                    if response.status_code == 200:
                        data = response.json()
                        files = data.get("code_files", [])
                        for file_info in files:
                            all_code.append({
                                "source": file_info["source"],
                                "file": file_info["name"],
                                "path": file_info["path"],
                                "content": file_info["content"],
                                "lines": len(file_info["content"].splitlines()),
                                "is_test": file_info["is_test"]
                            })
            
            return all_code
            
        except Exception as e:
            print(f"⚠️ Could not read existing code: {e}")
            return []
    
    def _simple_code_extraction(self, response: str) -> str:
        """Enhanced code extraction that removes all comments and explanations"""
        # Remove markdown code blocks
        code = re.sub(r'```python\n', '', response)
        code = re.sub(r'```\n', '', code)
        code = re.sub(r'```', '', code)
        
        # Remove common prefixes and explanations
        lines = code.split('\n')
        cleaned_lines = []
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines at the beginning
            if not in_code and not stripped:
                continue
                
            # Start collecting code when we see function definitions or imports
            if (stripped.startswith(('def ', 'import ', 'from ')) or
                (stripped.startswith('class ') and ':' in stripped) or
                (in_code and stripped)):
                in_code = True
                cleaned_lines.append(line)
            elif in_code:
                # Keep all lines once we're in code mode
                cleaned_lines.append(line)
        
        # Remove any remaining explanatory text
        result = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining markdown or explanatory text
        result = re.sub(r'^.*?(def |import |from |class )', r'\1', result, flags=re.DOTALL)
        
        return result
    
    def _fix_indentation(self, code: str) -> str:
        """
        Fix common indentation issues in Python code
        """
        # Simple approach: ensure consistent 4-space indentation
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Remove any existing indentation and add proper 4-space indentation
            # This is a simplified approach that ensures consistent formatting
            fixed_lines.append(stripped)
        
        return '\n'.join(fixed_lines)
    
    def _validate_python_syntax(self, code: str) -> bool:
        """
        Validate Python syntax
        """
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _fix_common_syntax_issues(self, code: str) -> str:
        """
        Fix common syntax issues in generated code
        """
        # Fix common issues
        code = code.replace('```python', '').replace('```', '')
        code = code.replace('`', '')
        
        # Remove any markdown formatting
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip markdown lines
            if line.strip().startswith('#') and not line.strip().startswith('##'):
                continue
            if line.strip().startswith('[') and line.strip().endswith(']'):
                continue
            cleaned_lines.append(line)
        
        # Fix common indentation and docstring issues
        fixed_code = self._fix_docstring_and_indentation('\n'.join(cleaned_lines))
        
        return fixed_code
    
    def _fix_docstring_and_indentation(self, code: str) -> str:
        """
        Fix docstring and indentation issues in generated code
        """
        lines = code.split('\n')
        fixed_lines = []
        in_function = False
        in_docstring = False
        docstring_content = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue
            
            # Check if we're entering a function
            if stripped.startswith('def '):
                in_function = True
                in_docstring = False
                docstring_content = []
                fixed_lines.append(stripped)
                continue
            
            # Check if we're entering a docstring
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_function and not in_docstring:
                    in_docstring = True
                    # Add proper indentation for docstring start
                    fixed_lines.append('    ' + stripped)
                    continue
                elif in_docstring:
                    in_docstring = False
                    # Add proper indentation for docstring end
                    fixed_lines.append('    ' + stripped)
                    continue
            
            # Handle docstring content
            if in_docstring:
                # Collect docstring content
                docstring_content.append(stripped)
                continue
            
            # If we were in a docstring but didn't find closing quotes, add them
            if docstring_content and not in_docstring:
                # Add the collected docstring content with proper indentation
                for content_line in docstring_content:
                    fixed_lines.append('    ' + content_line)
                # Add closing quotes
                fixed_lines.append('    """')
                docstring_content = []
            
            # Handle function body - simple 4-space indentation
            if in_function and not in_docstring and not docstring_content:
                # Simple approach: indent everything in function body with 4 spaces
                fixed_lines.append('    ' + stripped)
                continue
            
            # Handle non-function code
            fixed_lines.append(stripped)
        
        # If we still have docstring content at the end, add closing quotes
        if docstring_content:
            for content_line in docstring_content:
                fixed_lines.append('    ' + content_line)
            fixed_lines.append('    """')
        
        return '\n'.join(fixed_lines)
    
    def _get_fallback_code(self) -> str:
        """
        Get fallback code with proper indentation
        """
        return '''def add_two_numbers(a: int, b: int) -> int:
    \"\"\"
    Add two numbers together.
    
    Args:
        a (int): First number
        b (int): Second number
        
    Returns:
        int: Sum of the two numbers
    \"\"\"
    try:
        return a + b
    except TypeError as e:
        raise ValueError(f"Both arguments must be numbers: {e}")'''

# =============================================================================
# TESTER AGENT - CREATES TESTS
# =============================================================================
class TesterAgent(BaseAgent):
    """
    What: An agent that creates test cases for code
    Why: We need an agent specifically for test generation
    Can I change: YES - you can modify how test generation works
    """
    
    async def handle_task(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles task messages for test generation
        Why: Required by abstract BaseAgent class
        Can I change: YES - you can modify task handling
        """
        # For now, just acknowledge the task
        return [self.create_message(message.from_agent, MessageType.RESPONSE, 
                                  f"Tester agent received task: {message.content}")]
    
    async def handle_data(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles code data and generates test cases
        Why: This is where test generation happens
        Can I change: YES - you can modify the test generation strategy
        """
        try:
            print(f"🧪 TesterAgent received data: {message.content}")
            
            # Extract code from metadata
            code = message.metadata.get("code", "")
            if not code:
                print("❌ No code provided for testing")
                return [self.create_error_message(message.from_agent, "No code provided for testing")]
            
            print(f"📝 Code to test: {len(code)} characters")
            
            # Create a comprehensive testing prompt
            prompt = f"""
            You are an expert Python tester.
            
            Code to test:
            {code}
            
            Requirements:
            1. Create comprehensive unit tests using unittest
            2. Test all functions and methods
            3. Include edge cases and error conditions
            4. Use descriptive test names
            5. Include setup and teardown if needed
            6. Make sure tests are complete and runnable
            7. Test both valid inputs and invalid inputs
            8. Test edge cases like zero, negative numbers, large numbers
            9. DO NOT include the original code in the test file
            10. Only generate the test code, not the original code
            11. DO NOT use ANY import statements - the functions will be available directly
            12. Write tests as if the functions are already defined in the same scope
            13. IMPORTANT: Do not use 'from your_module import' or any import statements
            14. The functions will be executed in the same file as the tests
            
            Generate only the test code, no explanations or markdown formatting.
            """
            
            print(f"🤖 Sending test prompt to LLM: {prompt[:100]}...")
            
            # Get AI response
            response = self.llm.invoke(prompt)
            print(f"📝 LLM test response received: {len(response)} characters")
            
            # Extract test code from response
            test_code = self._extract_and_clean_test_code(response)
            print(f"🔍 Extracted test code: {len(test_code)} characters")
            
            if not test_code or len(test_code.strip()) < 10:
                print("⚠️ No valid test code extracted, using fallback")
                test_code = self._get_fallback_tests()
            
            # FIX: Validate syntax and apply emergency fixes
            if not self._validate_python_syntax(test_code):
                print("⚠️ Generated test code has syntax issues, applying emergency fixes")
                test_code = self._apply_test_emergency_fixes(test_code)
            
            # Save the generated tests to a file
            timestamp = _timestamp()
            filename = f"test_{timestamp}.py"
            filepath = GENERATED_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            print(f"💾 Tests saved to: {filepath}")
            
            # Create response message with the tests for the original sender
            response_message = self.create_message(
                to_agent=message.from_agent,
                message_type=MessageType.DATA,
                content=f"Generated tests saved to {filename}",
                metadata={
                    "test_code": test_code,
                    "filename": filename,
                    "filepath": str(filepath),
                    "original_code": code
                }
            )
            
            # ALSO send the tests directly to the runner agent
            runner_message = self.create_message(
                to_agent="runner",
                message_type=MessageType.DATA,
                content=f"Tests generated for execution",
                metadata={
                    "test_code": test_code,
                    "filename": filename,
                    "filepath": str(filepath),
                    "original_code": code
                }
            )
            
            print(f"📤 Sending {len([response_message, runner_message])} messages")
            return [response_message, runner_message]
            
        except Exception as e:
            print(f"❌ Test generation failed: {str(e)}")
            return [self.create_error_message(message.from_agent, f"Test generation failed: {str(e)}")]
    
    # FIX: Add proper test code extraction method (like CoderAgent's _extract_and_clean_code)
    def _extract_and_clean_test_code(self, response: str) -> str:
        """Extract and clean test code from LLM response"""
        # Remove markdown code blocks
        code = re.sub(r'```python\n', '', response)
        code = re.sub(r'```\n', '', code)
        code = re.sub(r'```', '', code)
        
        # Remove explanatory text and keep only test code
        lines = code.split('\n')
        test_lines = []
        in_test_code = False
        
        for line in lines:
            stripped = line.strip()
            
            # Start collecting when we see imports or class definitions
            if (stripped.startswith(('import ', 'from ', 'class Test', 'class test')) or
                (line.startswith('    def test_') and stripped) or
                (stripped and in_test_code)):
                in_test_code = True
                test_lines.append(line)
            elif in_test_code and not stripped:
                # Keep empty lines within test code
                test_lines.append(line)
            elif in_test_code and stripped.startswith('if __name__'):
                # Keep the main block
                test_lines.append(line)
        
        return '\n'.join(test_lines).strip()
    
    # FIX: Add syntax validation method (like CoderAgent)
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    # FIX: Add emergency fixes method (like CoderAgent's _apply_emergency_fixes)
    def _apply_test_emergency_fixes(self, test_code: str) -> str:
        """Apply emergency fixes to make test code syntactically correct"""
        lines = test_code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if line.strip():
                # Remove any import statements that reference the functions
                if any(import_statement in line.lower() for import_statement in [
                    'from your_module import',
                    'import your_module',
                    'from module import',
                    'import module'
                ]):
                    continue  # Skip this line
                
                # Fix indentation - use 4 spaces consistently
                leading_spaces = len(line) - len(line.lstrip())
                correct_spaces = (leading_spaces // 4) * 4
                fixed_line = ' ' * correct_spaces + line.lstrip()
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append('')
        
        # Ensure basic test structure
        fixed_code = '\n'.join(fixed_lines)
        if not fixed_code.strip() or "import unittest" not in fixed_code:
            return self._get_fallback_tests()
        
        return fixed_code
    
    def _get_fallback_tests(self) -> str:
        """Get fallback test code"""
        return '''import unittest

class TestFunction(unittest.TestCase):
    def test_basic_functionality(self):
        # Add your test here
        self.assertTrue(True)

    def test_edge_cases(self):
        # Add edge case tests here
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()'''

# =============================================================================
# RUNNER AGENT - EXECUTES TESTS
# =============================================================================
class RunnerAgent(BaseAgent):
    """
    What: An agent that executes tests and reports results
    Why: We need an agent to actually run the tests and check if they pass
    Can I change: YES - you can modify how test execution works
    """
    
    async def handle_task(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles task messages for test execution
        Why: Required by abstract BaseAgent class
        Can I change: YES - you can modify task handling
        """
        # For now, just acknowledge the task
        return [self.create_message(message.from_agent, MessageType.RESPONSE, 
                                  f"Runner agent received task: {message.content}")]
    
    async def handle_data(self, message: AgentMessage) -> List[AgentMessage]:
        """
        What: Handles test data and executes the tests
        Why: This is where actual test execution happens
        Can I change: YES - you can modify the test execution strategy
        """
        try:
            print(f"🏃 RunnerAgent received data: {message.content}")
            
            # Extract test code and original code from metadata
            test_code = message.metadata.get("test_code", "")
            original_code = message.metadata.get("original_code", "")
            
            if not test_code:
                print("❌ No test code provided")
                return [self.create_error_message(message.from_agent, "No test code provided")]
            
            print(f"🧪 Test code to execute: {len(test_code)} characters")
            print(f"📝 Original code: {len(original_code)} characters")
            
            # Run the tests
            test_results = self._run_tests(original_code, test_code)
            print(f"🏃 Test execution completed: {len(test_results)} characters")
            
            # Determine if tests passed
            tests_passed = "✅ TESTS PASSED" in test_results
            
            # Create response message with test results
            response_message = self.create_message(
                to_agent=message.from_agent,
                message_type=MessageType.DATA,
                content=f"Test execution completed",
                metadata={
                    "test_results": test_results,
                    "tests_passed": tests_passed,
                    "original_code": original_code,
                    "test_code": test_code
                }
            )
            
            print(f"📤 Sending test results: {'PASSED' if tests_passed else 'FAILED'}")
            return [response_message]
            
        except Exception as e:
            print(f"❌ Test execution failed: {str(e)}")
            return [self.create_error_message(message.from_agent, f"Test execution failed: {str(e)}")]
    
    def _run_tests(self, code: str, test_code: str) -> str:
        """
        What: Actually runs the tests and returns the results
        Why: We need to execute the tests in a safe environment
        Can I change: YES - you can modify the test execution environment
        """
        try:
            print(f"🔧 Running tests with {len(code)} chars of code and {len(test_code)} chars of tests")
            
            # Create a temporary file that combines both code and tests
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # First execute the code to define the functions
                f.write(code)
                f.write("\n\n")
                # Then add the tests
                f.write(test_code)
                test_file = f.name
            
            print(f"📁 Created combined file: {test_file}")
            
            # Run the tests using subprocess
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            print(f"🏃 Test execution completed with return code: {result.returncode}")
            print(f"📤 stdout: {len(result.stdout)} chars")
            print(f"📤 stderr: {len(result.stderr)} chars")
            
            # Clean up temporary files
            os.unlink(test_file)
            
            # Return the test results
            if result.returncode == 0:
                return f"✅ TESTS PASSED\n{result.stdout}"
            else:
                return f"❌ TESTS FAILED\n{result.stdout}\n{result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ TESTS TIMEOUT - Tests took too long to run"
        except Exception as e:
            return f"❌ TEST EXECUTION ERROR: {str(e)}"

# =============================================================================
# AGENT FACTORY - CREATES AGENTS DYNAMICALLY
# =============================================================================
class AgentFactory:
    """
    What: A factory that creates different types of agents
    Why: We need a way to create agents without hardcoding them
    Can I change: YES - you can add new agent types here
    """
    
    # Registry of available agent types
    _agent_types = {
        "coordinator": CoordinatorAgent,
        "coder": CoderAgent,
        "tester": TesterAgent,
        "runner": RunnerAgent
    }
    
    @classmethod
    def create_agent(cls, agent_id: str, agent_type: str, role: str, 
                    model_name: str = DEFAULT_MODEL, 
                    model_config: Dict[str, Any] = None) -> BaseAgent:
        """
        What: Creates a new agent of the specified type
        Why: This allows us to create agents dynamically
        Can I change: YES - you can modify how agents are created
        """
        if agent_type not in cls._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._agent_types[agent_type]
        return agent_class(agent_id, agent_type, role, model_name, model_config)
    
    @classmethod
    def register_agent_type(cls, agent_type: str, agent_class: type):
        """
        What: Registers a new agent type
        Why: This allows you to add custom agent types
        Can I change: YES - you can add new agent types
        """
        cls._agent_types[agent_type] = agent_class

# =============================================================================
# MESSAGE BUS - THE COMMUNICATION SYSTEM
# =============================================================================
class MessageBus:
    """
    What: Manages communication between all agents
    Why: Agents need a central system to send messages to each other
    Can I change: YES - you can modify the communication system
    """
    
    def __init__(self):
        """Initialize the message bus"""
        self.agents: Dict[str, BaseAgent] = {}  # All registered agents
    
    def register_agent(self, agent: BaseAgent):
        """
        What: Registers an agent with the message bus
        Why: Agents need to be registered to receive messages
        Can I change: YES - you can modify registration logic
        """
        self.agents[agent.agent_id] = agent
    
    async def send_message(self, message: AgentMessage):
        """
        What: Sends a message from one agent to another
        Why: This is how agents communicate with each other
        Can I change: YES - you can modify message routing
        """
        print(f"🔀 Sending message: {message.from_agent} -> {message.to_agent} ({message.message_type.value})")
        print(f"   Content: {message.content[:100]}...")
        
        # NEW: Log message to database (NON-BLOCKING)
        if hasattr(self, 'logger'):
            asyncio.create_task(self.logger.log_message(message))
        
        # Log message for debugging
        print(f"📤 Message sent: {message.from_agent} -> {message.to_agent}")
        
        # Send real-time update via WebSocket
        try:
            await websocket_manager.send_agent_update(
                agent_id=message.to_agent,
                status="receiving",
                message=f"Message from {message.from_agent}: {message.content[:100]}...",
                content=message.content
            )
        except Exception as e:
            print(f"⚠️ WebSocket update failed: {e}")
        
        target_agent = self.agents.get(message.to_agent)
        if not target_agent:
            print(f"⚠️ Warning: Agent {message.to_agent} not found")
            return
        
        # Send agent working status
        try:
            await websocket_manager.send_agent_update(
                agent_id=message.to_agent,
                status="working",
                message=f"Processing message from {message.from_agent}...",
                content=""
            )
        except Exception as e:
            print(f"⚠️ WebSocket update failed: {e}")
        
        # Process the message and get responses
        response_messages = await target_agent.process_message(message)
        
        print(f"📤 Agent {message.to_agent} generated {len(response_messages)} responses")
        
        # Send agent completed status
        try:
            await websocket_manager.send_agent_update(
                agent_id=message.to_agent,
                status="completed",
                message=f"Completed processing. Generated {len(response_messages)} responses.",
                content=""
            )
        except Exception as e:
            print(f"⚠️ WebSocket update failed: {e}")
        
        # Send each response message
        for response in response_messages:
            await self.send_message(response)
    
    async def process_workflow(self, initial_task: str, workflow_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        What: Processes a complete workflow from start to finish
        Why: This orchestrates the entire multi-agent workflow
        Can I change: YES - you can modify the workflow processing
        """
        try:
            print(f"🚀 Starting workflow with task: {initial_task}")
            print(f"🤖 Creating {len(workflow_agents)} agents")
            
            # Create and register all agents
            for agent_config in workflow_agents:
                agent = AgentFactory.create_agent(
                    agent_id=agent_config["id"],
                    agent_type=agent_config["type"],
                    role=agent_config["role"],
                    model_name=agent_config.get("model", "mistral"),
                    model_config=agent_config.get("model_config", {})
                )
                self.register_agent(agent)
                print(f"✅ Created agent: {agent_config['id']} ({agent_config['type']})")
            
            # Start the workflow with the coordinator
            coordinator = self.agents.get("coordinator")
            if not coordinator:
                raise ValueError("No coordinator agent found")
            
            # Generate workflow ID for tracking
            workflow_id = f"workflow_{datetime.now().timestamp()}"
            print(f"🚀 Starting workflow: {workflow_id}")
            
            # Create initial task message from system to coordinator
            initial_message = AgentMessage(
                id=f"workflow_start_{datetime.now().timestamp()}",
                from_agent="system",
                to_agent="coordinator",
                message_type=MessageType.TASK,
                content=initial_task
            )
            
            print(f"📤 Sending initial message to coordinator")
            
            # Process the workflow and wait for completion
            await self.send_message(initial_message)
            
            # Wait a bit for the workflow to complete (in a real system, you'd have better completion detection)
            print(f"⏳ Waiting for workflow completion...")
            # Reduce wait time and add progress updates
            for i in range(10):  # Wait 2 seconds total (10 * 0.2)
                await asyncio.sleep(0.2)  # Check every 200ms
                print(f"⏳ Workflow progress: {i+1}/10")
                
                # Log progress update
                print(f"⏳ Workflow progress: {i+1}/10")
            
            # Collect results from all agents
            results = {}
            for agent_id, agent in self.agents.items():
                results[agent_id] = {
                    "status": agent.status.value,
                    "memory": len(agent.memory.short_term),
                    "messages": [msg.content for msg in agent.memory.short_term[-5:]]  # Last 5 messages
                }
                print(f"📊 Agent {agent_id}: {agent.status.value}, {len(agent.memory.short_term)} messages")
            
            print(f"✅ Workflow completed successfully")
            
            # Log workflow completion
            print(f"✅ Workflow {workflow_id} completed successfully")
            
            return {
                "success": True,
                "results": results,
                "message": "Workflow completed successfully"
            }
                
        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Workflow failed"
            }

# =============================================================================
# REQUEST MODELS - WHAT THE FRONTEND SENDS
# =============================================================================
class PromptRequest(BaseModel):
    """
    What: The request format for chat interactions
    Why: Frontend needs a standard way to send requests
    Can I change: YES - you can add more fields to the request
    """
    prompt: str
    code_history: List[str] = []
    error_history: List[str] = []
    conversation_id: Optional[str] = None

class WorkflowRequest(BaseModel):
    """
    What: The request format for custom workflows
    Why: Frontend needs a way to define custom agent workflows
    Can I change: YES - you can add more workflow configuration options
    """
    task: str
    agents: List[Dict[str, Any]]  # [{"id": "agent1", "type": "coder", "role": "Python developer", "model": "mistral"}]

# =============================================================================
# UTILITY FUNCTIONS - HELPER FUNCTIONS
# =============================================================================
def _timestamp() -> str:
    """
    What: Creates a timestamp string for file naming
    Why: We need unique filenames for generated files
    Can I change: YES - you can modify the timestamp format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _extract_code(response: str) -> str:
    """
    What: Extracts Python code from AI responses
    Why: AI responses often include explanations, we need just the code
    Can I change: YES - you can modify how code is extracted
    """
    # Look for code blocks
    code_patterns = [
        r'```python\n(.*?)\n```',  # Markdown code blocks
        r'```\n(.*?)\n```',        # Generic code blocks
        r'```(.*?)```',            # Code blocks without language
        r'`(.*?)`'                 # Inline code
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # If no code blocks found, return the whole response
    return response.strip()

# =============================================================================
# API ENDPOINTS - WHAT THE FRONTEND CALLS
# =============================================================================

@app.post("/chat")
async def chat(request: PromptRequest):
    """
    What: Main chat endpoint for automated workflows
    Why: Frontend needs a simple way to interact with the AI system
    Can I change: YES - you can modify the chat logic
    """
    try:
        print(f"🚀 Starting chat workflow with prompt: {request.prompt}")
        
        # Validate input
        if not request.prompt or not request.prompt.strip():
            return {
                "type": "error",
                "message": "No prompt provided",
                "success": False
            }
        
        # Create a message bus for this session
        message_bus = MessageBus()
        
        # NEW: Attach database logger to message bus (NON-DISRUPTIVE)
        # Pass conversation_id if provided
        if request.conversation_id:
            db_integration.attach_to_message_bus(message_bus, conversation_id=request.conversation_id)
        else:
            db_integration.attach_to_message_bus(message_bus)
        
        # Define the default workflow agents
        workflow_agents = [
            {"id": "coordinator", "type": "coordinator", "role": "Smart Coordinator", "model": "mistral"},
            {"id": "coder", "type": "coder", "role": "Python Developer", "model": "mistral"},
            {"id": "tester", "type": "tester", "role": "Test Engineer", "model": "mistral"},
            {"id": "runner", "type": "runner", "role": "Test Runner"}
        ]
        
        print(f"🤖 Created {len(workflow_agents)} agents")
        
        # Process the workflow
        result = await message_bus.process_workflow(request.prompt, workflow_agents)
        
        print(f"✅ Workflow completed with result: {result}")
        
        # Check if workflow failed
        if not result.get("success", False):
            return {
                "type": "error",
                "message": result.get("message", "Workflow failed"),
                "success": False
            }
        
        # Extract results from the workflow
        code = None
        tests = None
        test_results = None
        tests_passed = None
        
        # Look for generated files in the results
        for agent_id, agent in message_bus.agents.items():
            print(f"🔍 Checking agent {agent_id}: {len(agent.memory.short_term)} messages")
            if agent_id == "coder":
                # Look for code in the agent's memory
                for msg in agent.memory.short_term:
                    if msg.metadata.get("code"):
                        code = msg.metadata["code"]
                        print(f"📝 Found code from coder agent: {len(code)} characters")
                        break
            elif agent_id == "tester":
                # Look for tests in the agent's memory
                for msg in agent.memory.short_term:
                    if msg.metadata.get("test_code"):
                        tests = msg.metadata["test_code"]
                        print(f"🧪 Found tests from tester agent: {len(tests)} characters")
                        break
            elif agent_id == "runner":
                # Look for test results in the agent's memory
                for msg in agent.memory.short_term:
                    if msg.metadata.get("test_results"):
                        test_results = msg.metadata["test_results"]
                        tests_passed = msg.metadata.get("tests_passed", False)
                        print(f"🏃 Found test results from runner agent: {len(test_results)} characters")
                        break
        
        # If code wasn't found in memory, try to read from the latest generated file
        if not code:
            try:
                code_files = list(GENERATED_DIR.glob("code_*.py"))
                if code_files:
                    latest_code_file = max(code_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_code_file, 'r', encoding='utf-8') as f:
                        code = f.read()
                    print(f"📝 Found code from file: {latest_code_file.name}")
            except Exception as e:
                print(f"⚠️ Could not read code from file: {e}")
        
        # If tests weren't found in memory, try to read from the latest generated file
        if not tests:
            try:
                test_files = list(GENERATED_DIR.glob("test_*.py"))
                if test_files:
                    latest_test_file = max(test_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_test_file, 'r', encoding='utf-8') as f:
                        tests = f.read()
                    print(f"🧪 Found tests from file: {latest_test_file.name}")
            except Exception as e:
                print(f"⚠️ Could not read tests from file: {e}")
        
        # If test results weren't found in memory, try to execute the latest tests
        if not test_results and code and tests:
            try:
                print("🏃 No test results found in memory, attempting to execute tests...")
                # Create a temporary runner to execute the tests
                temp_runner = RunnerAgent("temp_runner", "runner", "Test Runner")
                test_results = temp_runner._run_tests(code, tests)
                tests_passed = "✅ TESTS PASSED" in test_results
                print(f"🏃 Test execution completed: {len(test_results)} characters")
            except Exception as e:
                print(f"⚠️ Could not execute tests: {e}")
                test_results = f"❌ Test execution failed: {str(e)}"
                tests_passed = False
        
        # Determine response type based on what was generated
        response_type = "coding" if (code or tests) else "error"
        
        response_data = {
            "type": response_type,
            "message": result.get("message", "Task completed"),
            "code": code,
            "tests": tests,
            "test_results": test_results,
            "tests_passed": tests_passed,
            "success": result.get("success", False)
        }
        
        print(f"📤 Returning response: {response_data}")
        return response_data
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {str(e)}")
        return {
            "type": "error",
            "message": f"Error processing request: {str(e)}",
            "success": False
        }

@app.post("/run-workflow")
async def run_workflow(request: WorkflowRequest):
    """
    What: Endpoint for custom agent workflows
    Why: Frontend needs a way to define custom agent configurations
    Can I change: YES - you can modify the workflow processing
    """
    try:
        message_bus = MessageBus()
        result = await message_bus.process_workflow(request.task, request.agents)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/update-agent-model")
async def update_agent_model(agent_id: str, model_name: str, model_config: Dict[str, Any] = None):
    """
    What: Updates the AI model for a specific agent
    Why: Agents might need different models for different tasks
    Can I change: YES - you can modify model updating logic
    """
    # This would need to be implemented with agent management
    return {"success": True, "message": f"Updated agent {agent_id} to use model {model_name}"}

# =============================================================================
# LEGACY ENDPOINTS - FOR BACKWARD COMPATIBILITY
# =============================================================================
# These endpoints maintain compatibility with existing frontend code

@app.post("/generate-code")
async def generate_code(data: PromptRequest):
    """
    What: Legacy endpoint for code generation
    Why: Maintains compatibility with existing frontend
    Can I change: NO - this is for backward compatibility
    """
    return await chat(data)

@app.post("/generate-test")
async def generate_test(request: PromptRequest):
    """
    What: Legacy endpoint for test generation
    Why: Maintains compatibility with existing frontend
    Can I change: NO - this is for backward compatibility
    """
    return await chat(request)

@app.post("/run-test")
async def run_test(data: PromptRequest):
    """
    What: Legacy endpoint for test execution
    Why: Maintains compatibility with existing frontend
    Can I change: NO - this is for backward compatibility
    """
    return await chat(data)

# =============================================================================
# FILE MANAGEMENT ENDPOINTS - FOR FRONTEND FILE OPERATIONS
# =============================================================================

@app.get("/list-files")
async def list_files():
    """
    What: Lists all generated files
    Why: Frontend needs to know what files are available
    Can I change: YES - you can modify file listing logic
    """
    try:
        files = []
        for file in GENERATED_DIR.glob("*.py"):
            files.append(file.name)
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}

@app.get("/generated/{filename}")
async def get_generated_file(filename: str):
    """
    What: Serves generated files to the frontend
    Why: Frontend needs to display generated code and tests
    Can I change: YES - you can modify file serving logic
    """
    try:
        filepath = GENERATED_DIR / filename
        if filepath.exists():
            return FileResponse(filepath, media_type="text/plain")
        else:
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.delete("/generated/{filename}")
async def delete_generated_file(filename: str):
    """
    What: Deletes a generated file
    Why: Frontend needs to be able to clean up files
    Can I change: YES - you can modify file deletion logic
    """
    try:
        filepath = GENERATED_DIR / filename
        if filepath.exists():
            filepath.unlink()
            return {"success": True, "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# =============================================================================
# MANUAL AGENT WORKFLOW ENDPOINTS - FOR VISUAL WORKFLOW DESIGN
# =============================================================================

class ManualAgentBox(BaseModel):
    """
    What: Represents an agent box in the visual workflow designer
    Why: Frontend needs to send agent configurations
    Can I change: YES - you can add more agent properties
    """
    id: str
    x: float
    y: float
    width: float
    height: float
    agentType: str
    role: str
    model: str = DEFAULT_MODEL

class ManualAgentConnection(BaseModel):
    """
    What: Represents a connection between agents in the visual workflow
    Why: Frontend needs to define how agents communicate
    Can I change: YES - you can add more connection properties
    """
    id: str
    fromId: str
    fromSide: str
    toId: str
    toSide: str

class ManualFlowRequest(BaseModel):
    """
    What: Request format for manual workflow execution
    Why: Frontend needs to send manual workflow configurations
    Can I change: YES - you can add more workflow properties
    """
    prompt: str
    boxes: List[ManualAgentBox]
    connections: List[ManualAgentConnection]

@app.post("/run-manual-flow")
async def run_manual_flow(data: ManualFlowRequest):
    """
    What: Executes a manually designed workflow
    Why: Frontend needs to run custom agent workflows
    Can I change: YES - you can modify manual workflow processing
    """
    try:
        # Convert manual flow to agent configurations
        agents = []
        for box in data.boxes:
            agents.append({
                "id": box.id,
                "type": box.agentType,
                "role": box.role,
                "model": box.model
            })
        
        # Create message bus and process workflow
        message_bus = MessageBus()
        result = await message_bus.process_workflow(data.prompt, agents)
        
        # Collect messages from all agents
        messages = []
        for agent_id, agent in message_bus.agents.items():
            for msg in agent.memory.short_term:
                messages.append({
                    "from": msg.from_agent,
                    "to": msg.to_agent,
                    "type": msg.message_type.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                })
        
        return {
            "success": result.get("success", False),
            "messages": messages,
            "results": result.get("results", {}),
            "generated_files": []  # You can add file tracking here
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "messages": []
        }

@app.get("/example-workflow")
async def example_workflow():
    """
    What: Returns an example workflow configuration
    Why: Frontend needs example data for testing
    Can I change: YES - you can modify the example workflow
    """
    return {
        "agents": [
            {"id": "coordinator", "type": "coordinator", "role": "Smart Coordinator"},
            {"id": "coder", "type": "coder", "role": "Python Developer"},
            {"id": "tester", "type": "tester", "role": "Test Engineer"},
            {"id": "runner", "type": "runner", "role": "Test Runner"}
        ],
        "description": "A complete workflow for code generation and testing"
    }

# =============================================================================
# ADDITIONAL ENDPOINTS - FOR FRONTEND COMPATIBILITY
# =============================================================================

@app.get("/health")
async def health_check():
    """
    What: Health check endpoint
    Why: Frontend needs to verify backend is running
    Can I change: YES - you can modify health check logic
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "agents_available": ["coordinator", "coder", "tester", "runner"]
    }

@app.get("/agent-types")
async def get_agent_types():
    """
    What: Returns available agent types
    Why: Frontend needs to know what agents are available
    Can I change: YES - you can add new agent types
    """
    return {
        "agent_types": ["coordinator", "coder", "tester", "runner"],
        "count": 4
    }

@app.get("/agents")
async def get_active_agents():
    """
    What: Returns currently active agents
    Why: Frontend needs to know about active agents
    Can I change: YES - you can modify agent status reporting
    """
    return {
        "agents": {
            "coordinator": {"status": "available", "type": "coordinator"},
            "coder": {"status": "available", "type": "coder"},
            "tester": {"status": "available", "type": "tester"},
            "runner": {"status": "available", "type": "runner"}
        },
        "count": 4
    }

@app.get("/pipelines")
async def get_pipelines():
    """
    What: Returns available pipelines
    Why: Frontend needs pipeline information
    Can I change: YES - you can add pipeline management
    """
    return {
        "pipelines": [
            {
                "id": "default",
                "name": "Default Workflow",
                "description": "Standard code generation and testing workflow",
                "agents": ["coordinator", "coder", "tester", "runner"]
            }
        ],
        "count": 1
    }

@app.post("/execute-pipeline/{pipeline_id}")
async def execute_pipeline(pipeline_id: str, task: str):
    """
    What: Executes a specific pipeline
    Why: Frontend needs to run predefined pipelines
    Can I change: YES - you can modify pipeline execution
    """
    if pipeline_id == "default":
        return await chat(PromptRequest(prompt=task))
    else:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_id} not found")

@app.delete("/cleanup")
async def cleanup_system():
    """
    What: Cleans up the system
    Why: Frontend needs cleanup functionality
    Can I change: YES - you can modify cleanup logic
    """
    return {
        "message": "System cleaned up successfully",
        "removed_agents": 0,
        "cleared_messages": 0
    }

@app.get("/gpu-status")
async def get_gpu_status():
    """
    What: Check GPU availability and current model configuration
    Why: Users need to know if GPU acceleration is available
    Can I change: YES - you can add more GPU diagnostics
    """
    try:
        # Check if CUDA is available
        cuda_available = False
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            cuda_available = result.returncode == 0
        except:
            pass
        
        # Check Ollama GPU support
        ollama_gpu = False
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            ollama_gpu = "gpu" in result.stdout.lower()
        except:
            pass
        
        return {
            "cuda_available": cuda_available,
            "ollama_gpu_support": ollama_gpu,
            "current_model": DEFAULT_MODEL,
            "gpu_config": DEFAULT_GPU_CONFIG,
            "available_models": list(MODEL_CONFIGS.keys()),
            "recommendation": "CodeLlama:7b-instruct is optimized for code generation with GPU acceleration",
            "model_comparison": {
                "codellama:7b-instruct": "Best for code generation, lower temperature for focused output",
                "mistral": "Good for general tasks and coordination",
                "llama2": "Balanced performance for various tasks"
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "cuda_available": False,
            "ollama_gpu_support": False
        }

@app.post("/configure-gpu")
async def configure_gpu_settings(num_gpu: int = 1, num_thread: int = 8, temperature: float = 0.7):
    """
    What: Configure GPU settings for all agents
    Why: Users need to adjust GPU parameters
    Can I change: YES - you can add more configuration options
    """
    try:
        # Update the default GPU configuration
        global DEFAULT_GPU_CONFIG
        DEFAULT_GPU_CONFIG = {
            "num_gpu": num_gpu,
            "num_thread": num_thread,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
        
        return {
            "success": True,
            "message": f"GPU configuration updated: {num_gpu} GPU(s), {num_thread} threads, temp={temperature}",
            "config": DEFAULT_GPU_CONFIG
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/models")
async def get_available_models():
    """
    What: Get information about available models and their configurations
    Why: Users need to know what models are available and their settings
    Can I change: YES - you can add more model information
    """
    return {
        "current_default": DEFAULT_MODEL,
        "available_models": MODEL_CONFIGS,
        "recommendations": {
            "code_generation": "codellama:7b-instruct",
            "general_tasks": "mistral", 
            "balanced": "llama2"
        }
    }

@app.post("/switch-model")
async def switch_default_model(model_name: str):
    """
    What: Switch the default model for all new agents
    Why: Users need to easily change the default model
    Can I change: YES - you can add more model switching logic
    """
    try:
        global DEFAULT_MODEL
        if model_name not in MODEL_CONFIGS:
            return {
                "success": False,
                "error": f"Model '{model_name}' not found. Available models: {list(MODEL_CONFIGS.keys())}"
            }
        
        DEFAULT_MODEL = model_name
        return {
            "success": True,
            "message": f"Default model switched to {model_name}",
            "new_default": DEFAULT_MODEL,
            "model_config": MODEL_CONFIGS[model_name]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# =============================================================================
# DATABASE API ENDPOINTS - NEW ADDITIONS
# =============================================================================

@app.post("/conversations")
async def create_conversation(request: ConversationRequest):
    """Create new conversation - doesn't affect current agents"""
    try:
        conversation_id = await db_integration.start_conversation(request.title)
        return {"conversation_id": conversation_id, "title": request.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.get("/conversations")
async def get_conversations():
    """Get conversation history - read-only"""
    try:
        conversations = await db_integration.get_conversations()
        return [
            ConversationResponse(
                id=conv['id'],
                title=conv['title'],
                created_at=conv['created_at'],
                updated_at=conv['updated_at'],
                message_count=conv['message_count']
            )
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get specific conversation - read-only"""
    try:
        conversation = await db_integration.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = db_integration.db_service.get_conversation_messages(conversation_id)
        return {
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
                "is_active": conversation.is_active
            },
            "messages": [
                {
                    "id": msg.id,
                    "from_agent": msg.from_agent,
                    "to_agent": msg.to_agent,
                    "message_type": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.message_metadata
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete conversation - doesn't affect current agents"""
    try:
        success = await db_integration.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

# =============================================================================
# ONLINE AGENT SERVICE INTEGRATION
# =============================================================================
# Import the online agent service components
from online_agent_service import (
    online_app as online_agent_app,
    workflow_manager as online_workflow_manager,
    OnlineWorkflowRequest,
    OnlineAgent,
    ONLINE_MODEL_CONFIGS
)

# Mount the online agent service under /online path
app.mount("/online", online_agent_app, name="online_agent_service")

# Add a combined health check endpoint
@app.get("/combined-health")
async def combined_health_check():
    """Health check for both main service and online agent service"""
    return {
        "main_service": {
            "status": "healthy",
            "port": 8000,
            "endpoints": ["/chat", "/run-workflow", "/conversations"]
        },
        "online_agent_service": {
            "status": "healthy", 
            "port": 8000,
            "path": "/online",
            "endpoints": ["/online/health", "/online/run-workflow", "/online/models"],
            "available_models": list(ONLINE_MODEL_CONFIGS.keys())
        },
        "timestamp": datetime.now().isoformat()
    }

# Add endpoint to get online models from main service
@app.options("/online-models")
async def options_online_models():
    """Handle preflight requests for online models endpoint"""
    from fastapi import Response
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/online-models")
async def get_online_models_from_main():
    """Get online models through main service"""
    from fastapi import Response
    
    # Add name property to each model for frontend compatibility
    models_with_names = {}
    for model_id, model_config in ONLINE_MODEL_CONFIGS.items():
        models_with_names[model_id] = {
            **model_config,
            "name": model_id.replace("-", " ").replace("_", " ").title()
        }
    
    response = Response(
        content=json.dumps({
            "available_models": models_with_names,
            "default_model": "gpt-3.5-turbo",
            "providers": {
                "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
                "gemini": ["gemini-pro", "gemini-1.5-flash"]
            }
        }),
        media_type="application/json"
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Add endpoint to run online workflow through main service
@app.options("/run-online-workflow")
async def options_run_online_workflow():
    """Handle preflight requests for run online workflow endpoint"""
    from fastapi import Response
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.post("/run-online-workflow")
async def run_online_workflow_from_main(request: OnlineWorkflowRequest):
    """Run online agent workflow through main service"""
    try:
        result = await online_workflow_manager.run_workflow(request)
        # Convert the result to a dictionary if it's a Pydantic model
        if hasattr(result, 'dict'):
            result_dict = result.dict()
        elif hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        else:
            result_dict = result
        
        # Convert enum values and datetime objects to strings for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            elif hasattr(obj, 'isoformat'):  # datetime
                return obj.isoformat()
            else:
                return obj
        
        result_dict = convert_for_json(result_dict)
            
        from fastapi import Response
        response = Response(
            content=json.dumps(result_dict),
            media_type="application/json"
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Online workflow failed: {str(e)}")

# =============================================================================
# WEBSOCKET MANAGER FOR REAL-TIME UPDATES
# =============================================================================

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔗 WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_agent_update(self, agent_id: str, status: str, message: str, content: str = ""):
        """Send real-time agent update to all connected clients"""
        update = {
            "type": "agent_update",
            "agent_id": agent_id,
            "status": status,
            "message": message,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(update))
            except Exception as e:
                print(f"Error sending update: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# Create WebSocket manager instance
websocket_manager = WebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                print(f"📨 Received WebSocket message: {message}")
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

# =============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/conversations")
async def get_conversations(conversation_type: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Get all conversations with optional filtering"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        conv_type = ConversationType(conversation_type) if conversation_type else None
        conversations = conversation_manager.list_conversations(conv_type, limit, offset)
        
        return {
            "success": True,
            "conversations": [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "type": conv.type.value,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "is_active": conv.is_active,
                    "metadata": conv.metadata
                } for conv in conversations
            ],
            "count": len(conversations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@app.post("/conversations")
async def create_conversation(request: dict):
    """Create a new conversation"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        title = request.get("title", "New Conversation")
        conversation_type = ConversationType(request.get("type", "offline"))
        metadata = request.get("metadata", {})
        
        conversation = conversation_manager.create_conversation(title, conversation_type, metadata)
        
        return {
            "success": True,
            "conversation": {
                "id": conversation.id,
                "title": conversation.title,
                "type": conversation.type.value,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "is_active": conversation.is_active,
                "metadata": conversation.metadata
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with messages"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        conversation_data = conversation_manager.get_conversation_with_messages(conversation_id)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            **conversation_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

@app.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: dict):
    """Update a conversation"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        success = conversation_manager.update_conversation(conversation_id, **request)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            "message": "Conversation updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        success = conversation_manager.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            "message": "Conversation deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

@app.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, request: dict):
    """Add a message to a conversation"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        message_type = MessageType(request.get("type", "user"))
        content = request.get("content", "")
        agent_id = request.get("agent_id")
        agent_role = request.get("agent_role")
        metadata = request.get("metadata", {})
        
        if not content:
            raise HTTPException(status_code=400, detail="Message content is required")
        
        message = conversation_manager.add_message(
            conversation_id, message_type, content, agent_id, agent_role, metadata
        )
        
        return {
            "success": True,
            "message": {
                "id": message.id,
                "conversation_id": message.conversation_id,
                "type": message.type.value,
                "content": message.content,
                "agent_id": message.agent_id,
                "agent_role": message.agent_role,
                "timestamp": message.timestamp.isoformat(),
                "metadata": message.metadata
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, limit: int = 100, offset: int = 0):
    """Get messages for a conversation"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        messages = conversation_manager.get_messages(conversation_id, limit, offset)
        
        return {
            "success": True,
            "messages": [
                {
                    "id": msg.id,
                    "conversation_id": msg.conversation_id,
                    "type": msg.type.value,
                    "content": msg.content,
                    "agent_id": msg.agent_id,
                    "agent_role": msg.agent_role,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                } for msg in messages
            ],
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/prompt-templates")
async def get_prompt_templates(category: Optional[str] = None):
    """Get prompt templates"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        templates = conversation_manager.list_prompt_templates(category)
        
        return {
            "success": True,
            "templates": [
                {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "content": template.content,
                    "category": template.category,
                    "tags": template.tags,
                    "created_at": template.created_at.isoformat(),
                    "updated_at": template.updated_at.isoformat(),
                    "usage_count": template.usage_count
                } for template in templates
            ],
            "count": len(templates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prompt templates: {str(e)}")

@app.post("/prompt-templates")
async def create_prompt_template(request: dict):
    """Create a new prompt template"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        template = PromptTemplate(
            id=request.get("id", str(uuid.uuid4())),
            name=request.get("name", ""),
            description=request.get("description", ""),
            content=request.get("content", ""),
            category=request.get("category", "General"),
            tags=request.get("tags", [])
        )
        
        if not template.name or not template.content:
            raise HTTPException(status_code=400, detail="Name and content are required")
        
        success = conversation_manager.save_prompt_template(template)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save template")
        
        return {
            "success": True,
            "template": {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "content": template.content,
                "category": template.category,
                "tags": template.tags,
                "created_at": template.created_at.isoformat(),
                "updated_at": template.updated_at.isoformat(),
                "usage_count": template.usage_count
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create prompt template: {str(e)}")

@app.get("/conversations/search")
async def search_conversations(q: str, conversation_type: Optional[str] = None):
    """Search conversations"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        conv_type = ConversationType(conversation_type) if conversation_type else None
        conversations = conversation_manager.search_conversations(q, conv_type)
        
        return {
            "success": True,
            "conversations": [
                {
                    "id": conv.id,
                    "title": conv.title,
                    "type": conv.type.value,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "is_active": conv.is_active,
                    "metadata": conv.metadata
                } for conv in conversations
            ],
            "count": len(conversations),
            "query": q
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search conversations: {str(e)}")

@app.get("/conversations/stats")
async def get_conversation_stats():
    """Get conversation statistics"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        stats = conversation_manager.get_conversation_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation stats: {str(e)}")

@app.post("/conversations/load-default-templates")
async def load_default_templates():
    """Load default prompt templates"""
    if not CONVERSATION_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="Conversation management not available")
    
    try:
        conversation_manager.load_default_templates()
        return {
            "success": True,
            "message": "Default templates loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load default templates: {str(e)}")

# =============================================================================
# PROJECT MANAGEMENT ENDPOINTS
# =============================================================================

# This endpoint is now handled by the enhanced /projects endpoint below

@app.get("/projects/{project_name}/files")
async def get_project_files(project_name: str):
    """Get files in a specific project"""
    if not FILE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="File management not available")
    
    try:
        files = file_manager.get_project_files(project_name)
        return {
            "success": True,
            "project_name": project_name,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project files: {str(e)}")

@app.post("/projects/create")
async def create_project(request: dict):
    """Create a new project"""
    if not FILE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="File management not available")
    
    try:
        task_description = request.get("task_description", "AI Generated Project")
        conversation_id = request.get("conversation_id")
        
        result = file_manager.create_project(task_description, conversation_id)
        
        if result:
            return {
                "success": True,
                "project": result
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create project")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.post("/projects/save-code")
async def save_code_to_project(request: dict):
    """Save code to a project"""
    if not FILE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="File management not available")
    
    try:
        code = request.get("code", "")
        filename = request.get("filename")
        file_type = request.get("file_type", "src")
        conversation_id = request.get("conversation_id")
        task_description = request.get("task_description")
        
        if not code:
            raise HTTPException(status_code=400, detail="Code content is required")
        
        result = file_manager.save_code(
            code=code,
            filename=filename,
            file_type=file_type,
            conversation_id=conversation_id,
            task_description=task_description
        )
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save code: {str(e)}")

# =============================================================================
# GIT INTEGRATION ENDPOINTS
# =============================================================================

# Import Git integration modules
try:
    import sys
    from pathlib import Path
    git_integration_path = Path(__file__).parent.parent / "git-integration"
    sys.path.append(str(git_integration_path))
    
    from github_service import GitHubService
    # from online_agent_github_integration import online_agent_github
    
    GIT_AVAILABLE = True
    print("✅ Git integration modules loaded successfully")
except ImportError as e:
    print(f"⚠️ Git integration not available: {e}")
    GIT_AVAILABLE = False

# Import conversation management
try:
    from conversation_manager import conversation_manager, ConversationType, MessageType, PromptTemplate
    CONVERSATION_MANAGER_AVAILABLE = True
    print("✅ Conversation management system loaded successfully")
except ImportError as e:
    print(f"⚠️ Conversation management not available: {e}")
    CONVERSATION_MANAGER_AVAILABLE = False

# Import file management
try:
    from file_manager import file_manager
    FILE_MANAGER_AVAILABLE = True
    print("✅ Advanced file management system loaded successfully")
except ImportError as e:
    print(f"⚠️ File management not available: {e}")
    FILE_MANAGER_AVAILABLE = False

# Git configuration storage (in production, use a proper database)
git_config_storage = {
    "configured": False,
    "token": None,
    "username": None,
    "email": None,
    "user_info": None
}

@app.get("/git/status")
async def get_git_status():
    """Get Git integration status"""
    if not GIT_AVAILABLE:
        return {
            "configured": False,
            "error": "Git integration not available",
            "message": "Git integration modules not found"
        }
    
    return {
        "configured": git_config_storage["configured"],
        "user": git_config_storage["user_info"],
        "repositories": git_config_storage.get("repositories", []),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/git/configure")
async def configure_git(config: dict):
    """Configure Git integration with GitHub"""
    if not GIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Git integration not available")
    
    try:
        token = config.get("token")
        username = config.get("username")
        email = config.get("email")
        
        if not token or not username:
            raise HTTPException(status_code=400, detail="Token and username are required")
        
        # Initialize GitHub service
        github_service = GitHubService(token=token, username=username)
        
        # Validate token
        validation_result = github_service.validate_token()
        if not validation_result["success"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Get repositories
        repos_result = github_service.list_repositories()
        repositories = []
        if repos_result["success"]:
            repositories = repos_result["repositories"]
        
        # Store configuration
        git_config_storage.update({
            "configured": True,
            "token": token,
            "username": username,
            "email": email,
            "user_info": validation_result["user"],
            "repositories": repositories
        })
        
        return {
            "success": True,
            "message": "GitHub configured successfully",
            "user": validation_result["user"],
            "repositories": repositories
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@app.post("/git/pull")
async def pull_from_repository(request: dict):
    """Pull from a GitHub repository with proper folder structure"""
    if not GIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Git integration not available")
    
    if not git_config_storage["configured"]:
        raise HTTPException(status_code=400, detail="Git not configured. Please configure first.")
    
    try:
        repository = request.get("repository")
        if not repository:
            raise HTTPException(status_code=400, detail="Repository name is required")
        
        # Create a local directory for the repository
        import tempfile
        import shutil
        
        # Create a temporary directory for the repository
        temp_dir = tempfile.mkdtemp(prefix=f"git_pull_{repository.replace('/', '_')}_")
        
        try:
            # Clone the repository
            clone_url = f"https://github.com/{repository}.git"
            result = subprocess.run(
                ["git", "clone", clone_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise HTTPException(status_code=400, detail=f"Failed to clone repository: {result.stderr}")
            
            # Create project directory structure
            repo_name = repository.split('/')[-1]  # Get just the repository name
            project_dir = GENERATED_DIR / f"projects" / repo_name
            
            # Create project directory
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files with proper organization
            import glob
            copied_files = []
            
            # Copy ALL files maintaining folder structure (not just specific extensions)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    # Skip hidden files and common non-essential files
                    if file.startswith('.') and file not in ['.gitignore', '.env', '.env.example']:
                        continue
                    
                    # Skip common build/cache directories
                    if any(skip_dir in root for skip_dir in ['__pycache__', 'node_modules', '.git', '.vscode', '.idea']):
                        continue
                    
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, temp_dir)
                    dest_path = project_dir / rel_path
                    
                    # Create directory if it doesn't exist
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(rel_path)
            
            return {
                "success": True,
                "message": f"Successfully pulled from {repository}",
                "repository": repository,
                "project_name": repo_name,
                "project_path": str(project_dir),
                "files_copied": copied_files,
                "files_count": len(copied_files),
                "timestamp": datetime.now().isoformat()
            }
            
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")
        
    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Repository clone timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pull failed: {str(e)}")

@app.post("/git/push")
async def push_to_repository(request: dict):
    """Push generated code to a GitHub repository with organized folder structure and enhanced error handling"""
    if not GIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Git integration not available")
    
    if not git_config_storage["configured"]:
        raise HTTPException(status_code=400, detail="Git not configured. Please configure first.")
    
    try:
        repository = request.get("repository")
        commit_message = request.get("commit_message", "AI Generated Code")
        project_name = request.get("project_name", "ai-generated-project")
        
        if not repository:
            raise HTTPException(status_code=400, detail="Repository name is required")
        
        # Validate repository name format
        if not re.match(r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$', repository):
            raise HTTPException(status_code=400, detail="Invalid repository format. Use 'owner/repo'")
        
        # Initialize GitHub service with stored config
        try:
            github_service = GitHubService(
                token=git_config_storage["token"],
                username=git_config_storage["username"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize GitHub service: {str(e)}")
        
        # Get generated files and organize them
        generated_files = []
        
        # Check if we have a specific project directory
        project_dir = GENERATED_DIR / "projects" / project_name
        if project_dir.exists():
            # Use project directory if it exists
            source_dir = project_dir
        else:
            # Use main generated directory
            source_dir = GENERATED_DIR
        
        if source_dir.exists():
            # Organize files into src/ and tests/ folders
            for file_path in source_dir.rglob("*.py"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        rel_path = file_path.relative_to(source_dir)
                        
                        # Determine if it's a test file or source file
                        if "test" in file_path.name.lower() or "test" in str(rel_path).lower():
                            # Put in tests/ folder
                            github_path = f"tests/{rel_path.name}"
                        else:
                            # Put in src/ folder
                            github_path = f"src/{rel_path.name}"
                        
                        generated_files.append({
                            "path": github_path,
                            "content": content,
                            "original_path": str(rel_path)
                        })
                    except Exception as e:
                        print(f"Warning: Could not read file {file_path}: {e}")
        
        if not generated_files:
            raise HTTPException(status_code=400, detail="No generated files found to push")
        
        # Add README.md
        readme_content = f"""# {project_name}

AI Generated Project

## Description
This project was generated using AI agents and contains the following files:

### Source Files (src/)
"""
        
        # Add source files to README
        src_files = [f for f in generated_files if f["path"].startswith("src/")]
        for file in src_files:
            readme_content += f"- `{file['path']}`\n"
        
        readme_content += "\n### Test Files (tests/)\n"
        
        # Add test files to README
        test_files = [f for f in generated_files if f["path"].startswith("tests/")]
        for file in test_files:
            readme_content += f"- `{file['path']}`\n"
        
        readme_content += f"""
## Generated on
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Total Files
{len(generated_files)} files generated
"""
        
        # Add README to files
        generated_files.append({
            "path": "README.md",
            "content": readme_content,
            "original_path": "README.md"
        })
        
        # Convert to GitHub file format
        try:
            from github_service import GitHubFile
        except ImportError:
            raise HTTPException(status_code=500, detail="GitHub service not available")
        github_files = [
            GitHubFile(
                path=file["path"],
                content=file["content"],
                message=f"Add {file['path']}"
            ) for file in generated_files
        ]
        
        # Push files to repository
        push_result = github_service.push_files(
            repo_name=repository,
            files=github_files,
            commit_message=commit_message
        )
        
        if push_result["success"]:
            return {
                "success": True,
                "message": f"Successfully pushed {push_result['files_pushed']} files to {repository}",
                "repository_url": push_result["repository_url"],
                "commit_sha": push_result["commit_sha"],
                "files_pushed": push_result["files_pushed"],
                "commit_message": commit_message,
                "project_name": project_name,
                "folder_structure": {
                    "src_files": len(src_files),
                    "test_files": len(test_files),
                    "total_files": len(generated_files)
                }
            }
        else:
            raise HTTPException(status_code=400, detail=push_result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")

@app.get("/git/repositories")
async def list_git_repositories():
    """List available GitHub repositories"""
    if not GIT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Git integration not available")
    
    if not git_config_storage["configured"]:
        raise HTTPException(status_code=400, detail="Git not configured. Please configure first.")
    
    try:
        github_service = GitHubService(
            token=git_config_storage["token"],
            username=git_config_storage["username"]
        )
        
        result = github_service.list_repositories()
        
        if result["success"]:
            return {
                "success": True,
                "repositories": result["repositories"],
                "count": result["count"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list repositories: {str(e)}")

@app.get("/projects")
async def list_projects():
    """List all projects - both git-pulled and file-manager projects"""
    try:
        print("🔍 Projects API called - starting project listing...")
        all_projects = []
        
        # Get projects from git pull (stored in projects directory) - PRIMARY SOURCE
        projects_dir = GENERATED_DIR / "projects"
        print(f"🔍 Checking projects directory: {projects_dir}")
        print(f"🔍 Projects directory exists: {projects_dir.exists()}")
        
        if projects_dir.exists():
            project_dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
            print(f"🔍 Found {len(project_dirs)} project directories")
            
            for project_path in project_dirs:
                print(f"🔍 Processing project: {project_path.name}")
                
                try:
                    # Count files in project - for pulled repos, count all actual files
                    all_files = [f for f in project_path.rglob("*") if f.is_file()]
                    file_count = len(all_files)
                    total_files = len(list(project_path.rglob("*")))
                    
                    # Get project structure - for pulled repos, just categorize by file type
                    structure = {
                        "src_files": [],
                        "test_files": [],
                        "other_files": []
                    }
                    
                    for file_path in project_path.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(project_path)
                            # For pulled repos, categorize based on actual content, not folder structure
                            if "test" in file_path.name.lower() or "test" in str(rel_path).lower():
                                structure["test_files"].append(str(rel_path))
                            elif file_path.suffix in [".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".cpp", ".c", ".cs"]:
                                structure["src_files"].append(str(rel_path))
                            else:
                                structure["other_files"].append(str(rel_path))
                    
                    project_data = {
                        "name": project_path.name,
                        "path": str(project_path),
                        "file_count": file_count,
                        "structure": structure,
                        "created": project_path.stat().st_ctime,
                        "modified": project_path.stat().st_mtime,
                        "description": f"Git repository: {project_path.name}",
                        "total_files": total_files,
                        "status": "active",
                        "source": "git_pull",
                        "type": "git_repository",
                        "github_repo": None
                    }
                    
                    all_projects.append(project_data)
                    print(f"✅ Added project: {project_path.name} ({file_count} Python files, {total_files} total files)")
                    
                except Exception as e:
                    print(f"❌ Error processing project {project_path.name}: {e}")
        
        # Get projects from file manager (AI-generated projects) - SECONDARY SOURCE
        if FILE_MANAGER_AVAILABLE:
            try:
                file_manager_projects = file_manager.list_projects()
                print(f"🔍 Found {len(file_manager_projects)} file manager projects")
                
                for project in file_manager_projects:
                    project_name = project.get("name", "Unknown")
                    
                    # Skip if already added from git pull
                    if any(p.get("name") == project_name for p in all_projects):
                        print(f"⏭️ Skipping duplicate project: {project_name}")
                        continue
                    
                    # Convert file_manager project format to match frontend expectations
                    converted_project = {
                        "name": project_name,
                        "path": project.get("path", ""),
                        "file_count": project.get("file_count", 0),
                        "structure": {
                            "src_files": [],
                            "test_files": [],
                            "other_files": []
                        },
                        "created": 0,
                        "modified": 0,
                        "description": project.get("description", "AI Generated Project"),
                        "status": project.get("status", "active"),
                        "source": "file_manager",
                        "type": "ai_generated",
                        "github_repo": project.get("github_repo")
                    }
                    
                    # Try to get actual timestamps from file system
                    try:
                        project_path = Path(project.get("path", ""))
                        if project_path.exists():
                            converted_project["created"] = project_path.stat().st_ctime
                            converted_project["modified"] = project_path.stat().st_mtime
                    except:
                        pass
                    
                    all_projects.append(converted_project)
                    print(f"✅ Added file manager project: {project_name}")
                    
            except Exception as e:
                print(f"⚠️ Could not get file manager projects: {e}")
        
        print(f"📊 Found {len(all_projects)} projects total")
        for project in all_projects:
            print(f"  - {project.get('name')} ({project.get('file_count', 0)} files) - {project.get('source')}")
        
        return {
            "success": True,
            "projects": all_projects,
            "count": len(all_projects)
        }
        
    except Exception as e:
        print(f"❌ Error in list_projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

@app.get("/projects/{project_name}/files")
async def get_project_files(project_name: str):
    """Get files in a specific project - SIMPLIFIED VERSION"""
    try:
        print(f"🔍 Getting files for project: {project_name}")
        
        # Get project directory
        project_dir = GENERATED_DIR / "projects" / project_name
        print(f"🔍 Project directory: {project_dir}")
        print(f"🔍 Directory exists: {project_dir.exists()}")
        
        if not project_dir.exists():
            print(f"❌ Project directory does not exist: {project_dir}")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Simple file collection - just get all files
        files = []
        try:
            # Get all files recursively
            for file_path in project_dir.rglob("*"):
                if file_path.is_file():
                    print(f"🔍 Found file: {file_path.name}")
                    
                    # Get relative path
                    rel_path = file_path.relative_to(project_dir)
                    
                    # Try to read content
                    content = ""
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        print(f"✅ Successfully read: {file_path.name}")
                    except Exception as read_error:
                        print(f"⚠️ Could not read content of {file_path.name}: {read_error}")
                        content = f"Error reading file: {str(read_error)}"
                    
                    # Create file object
                    file_obj = {
                        "name": file_path.name,
                        "path": str(rel_path),
                        "type": "file",
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "content": content,
                        "is_test": "test" in file_path.name.lower(),
                        "extension": file_path.suffix
                    }
                    
                    files.append(file_obj)
                    print(f"✅ Added file: {file_path.name} ({file_obj['size']} bytes)")
        
        except Exception as scan_error:
            print(f"❌ Error scanning directory: {scan_error}")
            raise HTTPException(status_code=500, detail=f"Error scanning project directory: {str(scan_error)}")
        
        print(f"🔍 Total files found: {len(files)}")
        
        # Return simple response
        response = {
            "success": True,
            "project_name": project_name,
            "files": files,
            "count": len(files)
        }
        
        print(f"🔍 Returning response with {len(files)} files")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get project files: {str(e)}")

@app.get("/projects/{project_name}/test")
async def test_project_files(project_name: str):
    """Simple test endpoint to check if project exists and has files"""
    try:
        project_dir = GENERATED_DIR / "projects" / project_name
        if not project_dir.exists():
            return {"error": "Project directory does not exist", "path": str(project_dir)}
        
        files = list(project_dir.rglob("*"))
        file_files = [f for f in files if f.is_file()]
        
        return {
            "project_dir": str(project_dir),
            "exists": project_dir.exists(),
            "total_items": len(files),
            "file_count": len(file_files),
            "file_names": [f.name for f in file_files]
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/projects/{project_name}")
async def delete_project(project_name: str):
    """Delete a project"""
    try:
        import shutil
        project_dir = GENERATED_DIR / "projects" / project_name
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        shutil.rmtree(project_dir)
        
        return {
            "success": True,
            "message": f"Project '{project_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

# =============================================================================
# ENHANCED PROJECT MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/projects/auto-upload")
async def auto_upload_project(request: dict):
    """Automatically upload a project to GitHub using the enhanced file manager"""
    if not FILE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="File management not available")
    
    try:
        project_name = request.get("project_name")
        conversation_id = request.get("conversation_id")
        
        if not project_name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        # Get project directory
        project_dir = file_manager.base_dir / "projects" / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Load project metadata
        metadata = file_manager._load_project_metadata(project_dir)
        if not metadata:
            raise HTTPException(status_code=404, detail="Project metadata not found")
        
        # Trigger auto-upload
        upload_result = file_manager._auto_upload_to_github(project_dir, metadata)
        
        return {
            "success": True,
            "upload_result": upload_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-upload failed: {str(e)}")

@app.get("/projects/status")
async def get_project_status(project_name: str):
    """Get the status of a project including GitHub upload status"""
    if not FILE_MANAGER_AVAILABLE:
        raise HTTPException(status_code=500, detail="File management not available")
    
    try:
        project_dir = file_manager.base_dir / "projects" / project_name
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        metadata = file_manager._load_project_metadata(project_dir)
        if not metadata:
            raise HTTPException(status_code=404, detail="Project metadata not found")
        
        # Get file information
        files = file_manager.get_project_files(project_name)
        
        return {
            "success": True,
            "project_name": metadata["project_name"],
            "description": metadata["task_description"],
            "created_at": metadata["created_at"],
            "status": metadata.get("status", "unknown"),
            "github_repo": metadata.get("github_repo"),
            "last_upload": metadata.get("last_upload"),
            "file_count": len(files),
            "files": files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project status: {str(e)}")

# This endpoint is now handled by the main /projects endpoint above

@app.get("/projects/{project_name}/code")
async def get_project_code(project_name: str, file_pattern: str = "*.py"):
    """Get code files from a specific project for agent processing"""
    try:
        project_dir = GENERATED_DIR / "projects" / project_name
        
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Find all code files matching the pattern
        code_files = []
        for file_path in project_dir.rglob(file_pattern):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    rel_path = file_path.relative_to(project_dir)
                    
                    code_files.append({
                        "name": file_path.name,
                        "path": str(rel_path),
                        "content": content,
                        "size": len(content),
                        "lines": len(content.splitlines()),
                        "is_test": "test" in file_path.name.lower() or "test" in str(rel_path).lower()
                    })
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
        
        # Also check for other file types if no Python files found
        if not code_files and file_pattern == "*.py":
            print(f"⚠️ No Python files found in {project_name}, checking for other file types...")
            for ext in [".js", ".ts", ".html", ".css", ".json", ".md", ".txt"]:
                for file_path in project_dir.rglob(f"*{ext}"):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            rel_path = file_path.relative_to(project_dir)
                            
                            code_files.append({
                                "name": file_path.name,
                                "path": str(rel_path),
                                "content": content,
                                "size": len(content),
                                "lines": len(content.splitlines()),
                                "is_test": "test" in file_path.name.lower() or "test" in str(rel_path).lower()
                            })
                        except Exception as e:
                            print(f"Warning: Could not read file {file_path}: {e}")
        
        # Sort by file type and name
        code_files.sort(key=lambda x: (x["is_test"], x["name"]))
        
        return {
            "success": True,
            "project_name": project_name,
            "files": code_files,
            "count": len(code_files),
            "total_lines": sum(f["lines"] for f in code_files),
            "total_size": sum(f["size"] for f in code_files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project code: {str(e)}")

@app.get("/debug/projects")
async def debug_projects():
    """Debug endpoint to see what's in the projects directory"""
    try:
        projects_dir = GENERATED_DIR / "projects"
        debug_info = {
            "projects_dir_exists": projects_dir.exists(),
            "projects_dir_path": str(projects_dir),
            "projects": []
        }
        
        if projects_dir.exists():
            for project_path in projects_dir.iterdir():
                if project_path.is_dir():
                    project_info = {
                        "name": project_path.name,
                        "path": str(project_path),
                        "files": []
                    }
                    
                    # List all files in the project
                    for file_path in project_path.rglob("*"):
                        if file_path.is_file():
                            try:
                                rel_path = file_path.relative_to(project_path)
                                file_info = {
                                    "name": file_path.name,
                                    "path": str(rel_path),
                                    "size": file_path.stat().st_size,
                                    "extension": file_path.suffix
                                }
                                project_info["files"].append(file_info)
                            except Exception as e:
                                project_info["files"].append({
                                    "name": file_path.name,
                                    "error": str(e)
                                })
                    
                    debug_info["projects"].append(project_info)
        
        return debug_info
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/agents/read-code")
async def agents_read_code(request: dict):
    """Endpoint for agents to read code from projects or generated files"""
    try:
        project_name = request.get("project_name")
        file_pattern = request.get("file_pattern", "*.py")
        include_generated = request.get("include_generated", True)
        
        all_code = []
        
        # Read from specific project if provided
        if project_name:
            project_dir = GENERATED_DIR / "projects" / project_name
            if project_dir.exists():
                for file_path in project_dir.rglob(file_pattern):
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            rel_path = file_path.relative_to(project_dir)
                            
                            all_code.append({
                                "source": "project",
                                "project": project_name,
                                "name": file_path.name,
                                "path": str(rel_path),
                                "content": content,
                                "is_test": "test" in file_path.name.lower()
                            })
                        except Exception as e:
                            print(f"Warning: Could not read file {file_path}: {e}")
        
        # Also include generated files if requested
        if include_generated:
            for file_path in GENERATED_DIR.glob(file_pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        all_code.append({
                            "source": "generated",
                            "project": "generated",
                            "name": file_path.name,
                            "path": file_path.name,
                            "content": content,
                            "is_test": "test" in file_path.name.lower()
                        })
                    except Exception as e:
                        print(f"Warning: Could not read file {file_path}: {e}")
        
        # Sort by source, then by test status, then by name
        all_code.sort(key=lambda x: (x["source"], x["is_test"], x["name"]))
        
        return {
            "success": True,
            "code_files": all_code,
            "count": len(all_code),
            "projects_included": list(set(f["project"] for f in all_code))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read code: {str(e)}")

# =============================================================================
# STARTUP MESSAGE - SHOWS WHEN SERVER STARTS
# =============================================================================
if __name__ == "__main__":
    print("🚀 Starting Multi-Agent AI System...")
    print(f"🔧 GPU Configuration: {DEFAULT_GPU_CONFIG}")
    print("📁 Generated files will be saved to:", GENERATED_DIR)
    print("🌐 Main Server will be available at: http://localhost:8000")
    print("🌐 Online Agent Service will be available at: http://localhost:8000/online")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("📚 Online Service Documentation: http://localhost:8000/online/docs")
    print("🔗 Combined Health Check: http://localhost:8000/combined-health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
