@echo off
echo ========================================
echo    AI Agent Backend Service Starter
echo ========================================

cd backend-ai

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo ========================================
    echo    SETUP REQUIRED: API Keys
    echo ========================================
    echo.
    echo Please create a .env file with your API keys:
    echo.
    echo 1. Copy env_example.txt to .env
    echo 2. Edit .env and add your API keys:
    echo    - OPENAI_API_KEY=your_key_here
    echo    - MISTRAL_API_KEY=your_key_here  
    echo    - GEMINI_API_KEY=your_key_here
    echo.
    echo Get API keys from:
    echo - OpenAI: https://platform.openai.com/api-keys
    echo - Mistral: https://console.mistral.ai/api-keys/
    echo - Gemini: https://makersuite.google.com/app/apikey
    echo.
    pause
    exit /b 1
)

echo API Keys loaded from .env file
echo Starting main.py...
python main.py

pause
