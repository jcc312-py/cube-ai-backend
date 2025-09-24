
# =============================================================================
# ONLINE AGENT SERVICE WITH LANGCHAIN INTEGRATION
# =============================================================================



 # This service provides online model integration for manual agent workflows
# It uses LangChain for conversation tracking and online models (OpenAI, Anthropic, etc.)

import os
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks

# Add git-integration to path for GitHub upload
sys.path.append(str(Path(__file__).parent.parent / "git-integration"))

# Try to import GitHub integration
try:
    from online_agent_github_integration import online_agent_github
    GITHUB_AVAILABLE = True
    logging.info("✅ GitHub integration available")
except ImportError as e:
    GITHUB_AVAILABLE = False
    logging.warning(f"⚠️ GitHub integration not available: {e}")

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# LangChain imports for online models and conversation tracking
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
# Try to import Gemini, fallback gracefully if not available
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Gemini integration not available. Install with: pip install langchain-google-genai google-generativeai")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Removed unused imports: RunnableWithMessageHistory, PromptTemplate, StreamingStdOutCallbackHandler, CallbackManager

# Database integration (reusing existing structure)
from database import SafeDatabaseIntegration, ConversationRequest, ConversationResponse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# API Keys - Set these in environment variables or .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configurations
ONLINE_MODEL_CONFIGS = {
    "gpt-4": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": True
    },
    "gpt-3.5-turbo": {
        "provider": "openai", 
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": True
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model": "gpt-4-turbo-preview",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": True
    },
    "mistral-large": {
        "provider": "mistral",
        "model": "mistral-large-latest",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": True
    },
    "mistral-medium": {
        "provider": "mistral",
        "model": "mistral-medium-latest",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": True
    },
    "mistral-small": {
        "provider": "mistral",
        "model": "mistral-small-latest",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": False
    },
    "gemini-pro": {
        "provider": "gemini",
        "model": "gemini-1.5-pro",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": False
    },
    "gemini-1.5-flash": {
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_tokens": 4000,
        "streaming": False
    }
}

# Default model
DEFAULT_ONLINE_MODEL = "gemini-pro" if GEMINI_AVAILABLE else "mistral-small"

# =============================================================================
# DATA MODELS
# =============================================================================

class MessageType(Enum):
    """Message types for agent communication"""
    TASK = "task"
    DATA = "data" 
    REQUEST = "request"
    RESPONSE = "response"
    COORDINATION = "coordination"
    ERROR = "error"
    STATUS = "status"
    REVIEW = "review"

class OnlineAgentMessage(BaseModel):
    """Message structure for online agent communication"""
    id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_id: Optional[str] = None

class OnlineAgent(BaseModel):
    """Online agent configuration"""
    id: str
    name: str
    role: str
    model: str = DEFAULT_ONLINE_MODEL
    system_prompt: str = ""
    memory_enabled: bool = True
    conversation_id: Optional[str] = None

class OnlineAgentStatus(Enum):
    """Agent status tracking"""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"

class OnlineWorkflowRequest(BaseModel):
    """Request for running online agent workflow"""
    task: str
    agents: List[OnlineAgent]
    conversation_id: Optional[str] = None
    enable_streaming: bool = True

class OnlineWorkflowResponse(BaseModel):
    """Response from online agent workflow"""
    workflow_id: str
    status: str
    agents: Dict[str, OnlineAgentStatus]
    message_history: List[OnlineAgentMessage]
    total_messages: int
    conversation_id: str

# =============================================================================
# LANGCHAIN AGENT MANAGER
# =============================================================================

class LangChainAgentManager:
    """Manages LangChain agents with conversation tracking"""
    
    def __init__(self):
        self.agents: Dict[str, 'OnlineAgentInstance'] = {}
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: Dict[str, List[OnlineAgentMessage]] = {}
        
    def create_agent(self, agent_config: OnlineAgent) -> 'OnlineAgentInstance':
        """Create a new LangChain agent instance"""
        agent = OnlineAgentInstance(agent_config)
        self.agents[agent_config.id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional['OnlineAgentInstance']:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def create_conversation_memory(self, conversation_id: str) -> Dict[str, Any]:
        """Create conversation memory for tracking"""
        memory = {
            "messages": [],
            "conversation_id": conversation_id
        }
        self.conversations[conversation_id] = memory
        return memory
    
    def add_message_to_history(self, workflow_id: str, message: OnlineAgentMessage):
        """Add message to workflow history"""
        if workflow_id not in self.workflow_history:
            self.workflow_history[workflow_id] = []
        self.workflow_history[workflow_id].append(message)

# =============================================================================
# ONLINE AGENT INSTANCE
# =============================================================================

class OnlineAgentInstance:
    """Individual online agent with LangChain integration"""
    
    def __init__(self, config: OnlineAgent):
        self.config = config
        self.status = OnlineAgentStatus.IDLE
        self.llm = self._create_llm()
        self.memory = []
        
        if config.memory_enabled:
            self.memory = []
    
    def _create_llm(self):
        """Create LangChain LLM based on configuration"""
        model_config = ONLINE_MODEL_CONFIGS.get(self.config.model, ONLINE_MODEL_CONFIGS[DEFAULT_ONLINE_MODEL])
        
        if model_config["provider"] == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            return ChatOpenAI(
                model=model_config["model"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                streaming=model_config["streaming"],
                api_key=OPENAI_API_KEY
            )
        
        elif model_config["provider"] == "mistral":
            if not MISTRAL_API_KEY:
                raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
            
            return ChatMistralAI(
                model=model_config["model"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                streaming=model_config["streaming"],
                api_key=MISTRAL_API_KEY
            )
        
        elif model_config["provider"] == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Gemini integration not available. Install with: pip install langchain-google-genai google-generativeai")
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            
            return ChatGoogleGenerativeAI(
                model=model_config["model"],
                temperature=model_config["temperature"],
                google_api_key=GEMINI_API_KEY
            )
        
        else:
            raise ValueError(f"Unsupported provider: {model_config['provider']}")
    
    async def process_message(self, message: OnlineAgentMessage, conversation_memory: Optional[Dict[str, Any]] = None) -> str:
        """Process incoming message and return response"""
        try:
            self.status = OnlineAgentStatus.WORKING
            
            # Prepare enhanced system prompt for coordination
            coordination_instructions = f"""
            IMPORTANT: You are {self.config.name}, a {self.config.role} in a multi-agent workflow.
            
            YOUR ROLE: {self.config.role}
            YOUR TASK: {self.config.system_prompt}
            
            WORKFLOW RULES:
            1. Stay within your role and expertise
            2. Complete your specific task completely
            3. When finished, clearly state what you've accomplished
            4. Don't invent new tasks or roles
            5. Don't continue the conversation beyond your responsibility
            
            SPECIFIC INSTRUCTIONS BY ROLE:
            - COORDINATOR: Create a plan, delegate tasks, and summarize results. When all tasks are delegated and results received, say "COORDINATION COMPLETE"
            - CODER: Write the requested code completely with proper formatting. Include:
              * Complete, runnable code
              * Proper imports and dependencies
              * Clear comments for complex logic
              * Example usage if applicable
              When code is finished, say "CODE COMPLETE:" followed by the complete code in a code block (```python ... ```)
            - TESTER: Validate the provided code. When testing is done, say "TESTING COMPLETE: [brief summary of results]"
            - RUNNER: Execute the code and report results. When execution is done, say "EXECUTION COMPLETE: [brief summary of results]"
            
            CURRENT MESSAGE: {message.content}
            
            RESPOND ONLY WITH:
            1. Your specific contribution based on your role
            2. Clear completion statement when done
            3. Nothing else - no greetings, no continuation
            """
            
            system_content = f"{coordination_instructions}"
            
            # Create messages for LangChain
            messages = [SystemMessage(content=system_content)]
            
            # Add conversation history if available
            if conversation_memory and "messages" in conversation_memory:
                messages.extend(conversation_memory["messages"])
            
            # Add current message with context
            message_context = f"Message from {message.from_agent}: {message.content}"
            messages.append(HumanMessage(content=message_context))
            
            # Get response from LLM
            response = await self.llm.agenerate([messages])
            response_content = response.generations[0][0].text
            
            # Add to memory
            if conversation_memory and "messages" in conversation_memory:
                conversation_memory["messages"].extend([
                    HumanMessage(content=message_context),
                    AIMessage(content=response_content)
                ])
            
            # Save code to file if this is a coder agent and code is generated
            if "coder" in self.config.role.lower():
                # Check for various completion signals
                completion_signals = ["CODE COMPLETE:", "CODE COMPLETE", "```python", "```"]
                has_completion_signal = any(signal in response_content for signal in completion_signals)
                
                if has_completion_signal:
                    logging.info(f"🔍 Coder agent detected completion signal in response")
                    await self._save_generated_code(response_content, message.conversation_id)
                else:
                    logging.info(f"🔍 Coder agent response without completion signal: {response_content[:100]}...")
            
            self.status = OnlineAgentStatus.COMPLETED
            return response_content
            
        except Exception as e:
            self.status = OnlineAgentStatus.ERROR
            logging.error(f"Error processing message in agent {self.config.id}: {str(e)}")
            return f"Error: {str(e)}"
    
    async def _save_generated_code(self, response_content: str, conversation_id: str):
        """Save generated code to file and upload to GitHub"""
        try:
            # Import file manager
            from file_manager import file_manager
            
            # Extract code from response
            code = self._extract_code_from_response(response_content)
            if not code:
                logging.warning("No code found in response")
                return
            
            # Get task description from conversation context
            task_description = self._get_task_description_from_context(conversation_id)
            
            # Save code using advanced file manager
            result = file_manager.save_code(
                code=code,
                file_type="src",
                conversation_id=conversation_id,
                task_description=task_description
            )
            
            if result.get("success"):
                logging.info(f"💾 Code saved successfully: {result['filepath']}")
                logging.info(f"📁 Project: {result['project_name']}")
                
                # Log GitHub result
                github_result = result.get("github_result", {})
                if github_result.get("status") == "success":
                    logging.info(f"🐙 Uploaded to GitHub: {github_result.get('repo_url')}")
                elif github_result.get("status") == "github_not_available":
                    logging.info("ℹ️ GitHub not available - code saved locally only")
                else:
                    logging.warning(f"⚠️ GitHub upload issue: {github_result.get('error', 'Unknown error')}")
            else:
                logging.error(f"❌ Failed to save code: {result.get('error')}")
            
        except Exception as e:
            logging.error(f"❌ Error in _save_generated_code: {e}")
            # Fallback to simple file save
            try:
                code = self._extract_code_from_response(response_content)
                if code:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"code_{timestamp}.py"
                    filepath = Path("generated") / filename
                    filepath.parent.mkdir(exist_ok=True)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(code)
                    logging.info(f"💾 Fallback save to: {filepath}")
            except Exception as fallback_error:
                logging.error(f"❌ Fallback save also failed: {fallback_error}")
    
    def _get_task_description_from_context(self, conversation_id: str) -> str:
        """Extract task description from conversation context"""
        try:
            # This would ideally get the original task from conversation history
            # For now, return a generic description
            return "AI Generated Code from Online Agent Workflow"
        except Exception:
            return "AI Generated Code"
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from response"""
        try:
            logging.info(f"🔍 Extracting code from response: {response[:200]}...")
            
            # Look for code blocks with python specification
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end != -1:
                    code = response[start:end].strip()
                    logging.info(f"✅ Extracted Python code block: {len(code)} characters")
                    return code
            
            # Look for code blocks without language specification
            if "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    code = response[start:end].strip()
                    # Check if it looks like Python code
                    if any(keyword in code for keyword in ["def ", "import ", "class ", "if __name__", "print(", "return "]):
                        logging.info(f"✅ Extracted generic code block: {len(code)} characters")
                        return code
            
            # If no code blocks, check if the entire response is code
            if any(keyword in response for keyword in ["def ", "import ", "class ", "print(", "return "]):
                logging.info(f"✅ Extracted entire response as code: {len(response)} characters")
                return response.strip()
            
            logging.warning(f"❌ No code found in response")
            return ""
            
        except Exception as e:
            logging.error(f"Failed to extract code: {str(e)}")
            return ""
    
    def get_status(self) -> OnlineAgentStatus:
        """Get current agent status"""
        return self.status

# =============================================================================
# ONLINE WORKFLOW MANAGER
# =============================================================================

class OnlineWorkflowManager:
    """Manages online agent workflows with LangChain integration"""
    
    def __init__(self):
        self.agent_manager = LangChainAgentManager()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.db_integration = SafeDatabaseIntegration()
    
    async def run_workflow(self, request: OnlineWorkflowRequest) -> OnlineWorkflowResponse:
        """Run a complete workflow with online agents"""
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        
        # Create conversation if needed (only for non-manual flows)
        conversation_id = request.conversation_id
        if conversation_id is None:  # Explicitly None/undefined - don't create DB conversation
            conversation_id = f"manual_workflow_{workflow_id}"  # Use internal ID only
        elif not conversation_id:  # Empty string - create DB conversation
            conversation_id = await self.db_integration.start_conversation(f"Online Workflow: {request.task}")
        
        # Initialize workflow
        self.active_workflows[workflow_id] = {
            "status": "running",
            "agents": {},
            "message_history": [],
            "conversation_id": conversation_id
        }
        
        # Create agents
        agents = {}
        for agent_config in request.agents:
            agent = self.agent_manager.create_agent(agent_config)
            agents[agent_config.id] = agent
            self.active_workflows[workflow_id]["agents"][agent_config.id] = OnlineAgentStatus.IDLE
        
        # Create conversation memory
        conversation_memory = self.agent_manager.create_conversation_memory(conversation_id)
        
        # Start workflow execution
        try:
            # Find coordinator agent or use first agent
            coordinator = next((agent for agent in agents.values() if "coordinator" in agent.config.role.lower()), 
                             list(agents.values())[0])
            
            # Send initial task to coordinator
            initial_message = OnlineAgentMessage(
                from_agent="system",
                to_agent=coordinator.config.id,
                message_type=MessageType.TASK,
                content=request.task,
                conversation_id=conversation_id
            )
            
            # Process workflow
            await self._execute_workflow(workflow_id, agents, initial_message, conversation_memory)
            
            # Update final status
            self.active_workflows[workflow_id]["status"] = "completed"
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = "error"
            logging.error(f"Workflow error: {str(e)}")
        
        # Return response
        return OnlineWorkflowResponse(
            workflow_id=workflow_id,
            status=self.active_workflows[workflow_id]["status"],
            agents={agent_id: agent.get_status() for agent_id, agent in agents.items()},
            message_history=self.active_workflows[workflow_id]["message_history"],
            total_messages=len(self.active_workflows[workflow_id]["message_history"]),
            conversation_id=conversation_id
        )
    
    async def _execute_workflow(self, workflow_id: str, agents: Dict[str, OnlineAgentInstance], 
                              initial_message: OnlineAgentMessage, conversation_memory: Dict[str, Any]):
        """Execute the workflow step by step with multi-agent coordination"""
        current_message = initial_message
        max_iterations = 20
        iteration = 0
        agent_roles = {agent_id: agent.config.role.lower() for agent_id, agent in agents.items()}
        
        # Track agent completion status
        agent_completion = {agent_id: False for agent_id in agents.keys()}
        workflow_completed = False
        
        while iteration < max_iterations and not workflow_completed:
            # Add message to history
            self.active_workflows[workflow_id]["message_history"].append(current_message)
            self.agent_manager.add_message_to_history(workflow_id, current_message)
            
            # Save to database (only for non-manual workflows)
            if not current_message.conversation_id.startswith("manual_workflow_"):
                await self.db_integration.add_message_to_conversation(
                    current_message.conversation_id,
                    current_message.from_agent,
                    current_message.to_agent,
                    current_message.message_type.value,
                    current_message.content,
                    current_message.metadata
                )
            
            # Get target agent
            target_agent = agents.get(current_message.to_agent)
            if not target_agent:
                break
            
            # Update agent status
            self.active_workflows[workflow_id]["agents"][current_message.to_agent] = OnlineAgentStatus.WORKING
            
            # Process message
            response_content = await target_agent.process_message(current_message, conversation_memory)
            
            # Create response message and add to history
            response_message = OnlineAgentMessage(
                from_agent=current_message.to_agent,
                to_agent=current_message.from_agent,
                message_type=MessageType.RESPONSE,
                content=response_content,
                conversation_id=current_message.conversation_id
            )
            self.active_workflows[workflow_id]["message_history"].append(response_message)
            self.agent_manager.add_message_to_history(workflow_id, response_message)
            
            # Mark agent as completed for this iteration
            agent_completion[current_message.to_agent] = True
            
            # Check if workflow is complete based on agent roles and completion
            workflow_completed = self._check_workflow_completion(agent_roles, agent_completion, response_content)
            if workflow_completed:
                break
            
            # Find next agent based on workflow logic, not circular routing
            next_agent = self._get_next_agent(current_message.to_agent, agent_roles, agent_completion)
            if not next_agent:
                # No more agents to process
                workflow_completed = True
                break
            
            # Create message to next agent
            next_message = OnlineAgentMessage(
                from_agent=current_message.to_agent,
                to_agent=next_agent,
                message_type=MessageType.COORDINATION,
                content=f"Task completed by {current_message.to_agent}. Next step: {response_content}",
                conversation_id=current_message.conversation_id
            )
            current_message = next_message
            
            iteration += 1
        
        # Mark workflow as completed
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = "completed" if workflow_completed else "error"
    
    def _check_workflow_completion(self, agent_roles: Dict[str, str], agent_completion: Dict[str, bool], last_response: str) -> bool:
        """Check if workflow should be completed based on agent roles and responses"""
        # Check for explicit completion signals based on agent roles
        # Only accept completion signals from the appropriate agent type
        current_agent_role = None
        for agent_id, role in agent_roles.items():
            if agent_completion.get(agent_id, False):
                current_agent_role = role.lower()
                break
        
        if current_agent_role == "coordinator":
            # Coordinator can only complete with coordination-specific phrases
            if any(phrase in last_response.lower() for phrase in [
                "coordination complete", "workflow complete", "all tasks delegated"
            ]):
                return True
        elif current_agent_role == "coder":
            # Coder can complete with code-specific phrases
            if any(phrase in last_response.lower() for phrase in [
                "code complete", "implementation complete"
            ]):
                return True
        elif current_agent_role == "tester":
            # Tester can complete with testing-specific phrases
            if any(phrase in last_response.lower() for phrase in [
                "testing complete", "validation complete"
            ]):
                return True
        elif current_agent_role == "runner":
            # Runner can complete with execution-specific phrases
            if any(phrase in last_response.lower() for phrase in [
                "execution complete", "running complete"
            ]):
                return True
        
        # Check if all agents have completed their roles
        if all(agent_completion.values()):
            return True
        
        # For single agent workflows, only complete if the agent explicitly says it's done
        roles = list(agent_roles.values())
        if len(roles) == 1:
            # Single agent - only complete if agent explicitly indicates completion
            return any(phrase in last_response.lower() for phrase in [
                "code complete", "task complete", "done", "finished", "complete"
            ])
        
        # Check for specific role-based completion patterns for multi-agent workflows
        if len(roles) == 2 and "coordinator" in roles and "coder" in roles:
            # Simple coordinator + coder workflow
            # Only complete if both coordinator and coder have completed their tasks
            coordinator_completed = agent_completion.get("coordinator", False)
            coder_completed = agent_completion.get("coder", False)
            return coordinator_completed and coder_completed
        
        return False
    
    def _get_next_agent(self, current_agent: str, agent_roles: Dict[str, str], agent_completion: Dict[str, bool]) -> Optional[str]:
        """Get the next agent to process based on workflow logic, not circular routing"""
        current_role = agent_roles.get(current_agent, "").lower()
        
        # Define workflow patterns based on agent roles
        if current_role == "coordinator":
            # Coordinator should delegate to next available agent
            for agent_id, role in agent_roles.items():
                if agent_id != current_agent and not agent_completion.get(agent_id, False):
                    return agent_id
        
        elif current_role == "coder":
            # Coder completes code, workflow should end or go to tester if available
            for agent_id, role in agent_roles.items():
                if role == "tester" and not agent_completion.get(agent_id, False):
                    return agent_id
            # No tester, workflow complete
            return None
        
        elif current_role == "tester":
            # Tester completes validation, workflow should end
            return None
        
        elif current_role == "runner":
            # Runner executes code, workflow should end
            return None
        
        # Default: no more agents to process
        return None

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Create FastAPI app for online agent service
online_app = FastAPI(
    title="Online Agent Service",
    description="Online model integration for manual agent workflows with LangChain",
    version="1.0.0"
)

# Add CORS middleware
online_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow manager
workflow_manager = OnlineWorkflowManager()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@online_app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Online Agent Service",
        "version": "1.0.0",
        "description": "Online model integration with LangChain for manual agent workflows",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "workflow": "/run-workflow",
            "conversations": "/conversations",
            "workflow-status": "/workflow-status/{workflow_id}"
        }
    }

@online_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "online_agent_service",
        "timestamp": datetime.now().isoformat(),
        "available_models": list(ONLINE_MODEL_CONFIGS.keys()),
        "github_available": GITHUB_AVAILABLE
    }

@online_app.post("/upload-to-github")
async def upload_to_github():
    """Manually upload latest generated code to GitHub"""
    if not GITHUB_AVAILABLE:
        raise HTTPException(status_code=503, detail="GitHub integration not available")
    
    try:
        result = await online_agent_github.upload_latest_generated_code()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GitHub upload failed: {str(e)}")

@online_app.get("/github-status")
async def github_status():
    """Check GitHub integration status"""
    if not GITHUB_AVAILABLE:
        return {"available": False, "error": "GitHub integration not available"}
    
    try:
        is_configured = online_agent_github.is_configured()
        return {
            "available": True,
            "configured": is_configured,
            "message": "GitHub integration is ready" if is_configured else "GitHub not configured"
        }
    except Exception as e:
        return {"available": True, "configured": False, "error": str(e)}

@online_app.get("/models")
async def get_online_models():
    """Get available online models"""
    return {
        "available_models": ONLINE_MODEL_CONFIGS,
        "default_model": DEFAULT_ONLINE_MODEL,
        "providers": {
            "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
            "gemini": ["gemini-pro", "gemini-pro-vision"]
        }
    }

@online_app.post("/run-workflow")
async def run_online_workflow(request: OnlineWorkflowRequest):
    """Run online agent workflow"""
    # Validate request
    if not request.task or not request.task.strip():
        raise HTTPException(status_code=422, detail="Task cannot be empty")
    
    if not request.agents or len(request.agents) == 0:
        raise HTTPException(status_code=422, detail="At least one agent must be specified")
    
    # Check if required API keys are available
    required_providers = set()
    for agent in request.agents:
        model_config = ONLINE_MODEL_CONFIGS.get(agent.model, ONLINE_MODEL_CONFIGS[DEFAULT_ONLINE_MODEL])
        required_providers.add(model_config["provider"])
    
    missing_keys = []
    if "openai" in required_providers and not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if "mistral" in required_providers and not MISTRAL_API_KEY:
        missing_keys.append("MISTRAL_API_KEY")
    if "gemini" in required_providers and not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
    
    if missing_keys:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required API keys: {', '.join(missing_keys)}. Please set the environment variables."
        )
    
    try:
        response = await workflow_manager.run_workflow(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

@online_app.get("/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    workflow = workflow_manager.active_workflows.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "workflow_id": workflow_id,
        "status": workflow["status"],
        "agents": workflow["agents"],
        "message_count": len(workflow["message_history"]),
        "conversation_id": workflow["conversation_id"]
    }

@online_app.get("/conversations")
async def get_online_conversations():
    """Get conversation history"""
    try:
        conversations = await workflow_manager.db_integration.get_conversations()
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

@online_app.get("/conversations/{conversation_id}")
async def get_online_conversation(conversation_id: str):
    """Get specific conversation"""
    try:
        conversation = await workflow_manager.db_integration.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = workflow_manager.db_integration.db_service.get_conversation_messages(conversation_id)
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

# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Online Agent Service...")
    print("🔗 LangChain integration enabled")
    print("🌐 Online models available:", list(ONLINE_MODEL_CONFIGS.keys()))
    print("📚 API Documentation: http://localhost:8001/docs")
    print("⚠️  Make sure to set OPENAI_API_KEY, MISTRAL_API_KEY, and GEMINI_API_KEY environment variables")
    
    uvicorn.run(online_app, host="0.0.0.0", port=8001)

