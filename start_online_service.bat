@echo off
echo ========================================
echo   Online Agent Service Startup Script
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Check if API keys are set
echo Checking API keys...
if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  WARNING: OPENAI_API_KEY not set
    echo    You can still use Mistral models if MISTRAL_API_KEY is set
) else (
    echo ✅ OPENAI_API_KEY is set
)

if "%MISTRAL_API_KEY%"=="" (
    echo ⚠️  WARNING: MISTRAL_API_KEY not set
    echo    You can still use OpenAI models if OPENAI_API_KEY is set
) else (
    echo ✅ MISTRAL_API_KEY is set
)

echo.
echo Starting Online Agent Service...
echo 📚 API Documentation will be available at: http://localhost:8001/docs
echo 🌐 Service will be available at: http://localhost:8001
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start the service
python online_agent_service.py

pause
