# Online Agent Service Setup Guide

This guide will help you set up and use the Online Agent Service with OpenAI and Mistral AI integration.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+ installed
- Virtual environment activated
- API keys for OpenAI and/or Mistral AI

### 2. Installation

The dependencies are already installed in the virtual environment. If you need to reinstall:

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys Setup

You need to set environment variables for your API keys:

#### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
$env:MISTRAL_API_KEY="your-mistral-api-key-here"
```

#### Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
set MISTRAL_API_KEY=your-mistral-api-key-here
```

#### Linux/Mac:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export MISTRAL_API_KEY="your-mistral-api-key-here"
```

### 4. Starting the Service

#### Option 1: Using the batch file (Windows)
```bash
start_online_service.bat
```

#### Option 2: Manual startup
```bash
# Activate virtual environment
venv\Scripts\activate

# Start the service
python online_agent_service.py
```

The service will start on `http://localhost:8001`

## 📚 Available Models

### OpenAI Models
- `gpt-4` - GPT-4 model
- `gpt-3.5-turbo` - GPT-3.5 Turbo (default)
- `gpt-4-turbo` - GPT-4 Turbo Preview

### Mistral AI Models
- `mistral-large` - Mistral Large
- `mistral-medium` - Mistral Medium
- `mistral-small` - Mistral Small

## 🔧 API Endpoints

### Health Check
- `GET /health` - Service health status

### Models
- `GET /models` - List available models

### Workflows
- `POST /run-workflow` - Run an agent workflow
- `GET /workflow-status/{workflow_id}` - Get workflow status

### Conversations
- `GET /conversations` - List conversations
- `GET /conversations/{conversation_id}` - Get specific conversation

## 📖 API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python test_online_service.py
```

## 💡 Usage Examples

### Example 1: Simple Workflow

```python
from online_agent_service import OnlineAgent, OnlineWorkflowRequest

# Create agents
agents = [
    OnlineAgent(
        id="coder",
        name="Python Coder",
        role="programmer",
        model="gpt-3.5-turbo",
        system_prompt="You are a Python programmer."
    ),
    OnlineAgent(
        id="reviewer",
        name="Code Reviewer",
        role="reviewer",
        model="mistral-medium",
        system_prompt="You review code for quality and best practices."
    )
]

# Create workflow request
request = OnlineWorkflowRequest(
    task="Create a function to calculate fibonacci numbers",
    agents=agents
)
```

### Example 2: Using the API

```bash
# Get available models
curl http://localhost:8001/models

# Run a workflow
curl -X POST http://localhost:8001/run-workflow \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Create a simple calculator",
    "agents": [
      {
        "id": "coder",
        "name": "Programmer",
        "role": "programmer",
        "model": "gpt-3.5-turbo"
      }
    ]
  }'
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   - Make sure you're in the virtual environment
   - Run `pip install -r requirements.txt`

2. **API Key Errors**
   - Verify your API keys are set correctly
   - Check that you have credits/access to the models

3. **Port Already in Use**
   - Change the port in `online_agent_service.py` (line 548)
   - Or stop other services using port 8001

4. **Model Not Available**
   - Check if your API key has access to the requested model
   - Verify the model name is correct

### Getting Help

- Check the API documentation at http://localhost:8001/docs
- Review the test output from `test_online_service.py`
- Check the console output for error messages

## 🔄 Integration with Frontend

The Online Agent Service is designed to work with the existing frontend. The frontend can:

1. Connect to the service at `http://localhost:8001`
2. Use the `/run-workflow` endpoint to execute agent workflows
3. Monitor workflow status via `/workflow-status/{workflow_id}`
4. Access conversation history via `/conversations`

## 📝 Configuration

You can modify the service configuration in `online_agent_service.py`:

- **Default Model**: Change `DEFAULT_ONLINE_MODEL`
- **Model Configurations**: Modify `ONLINE_MODEL_CONFIGS`
- **Port**: Change the port in the `uvicorn.run()` call
- **CORS**: Update allowed origins in the CORS middleware

## 🚀 Next Steps

1. Set up your API keys
2. Start the service
3. Test with the provided examples
4. Integrate with your frontend application
5. Customize agent roles and prompts for your use case
