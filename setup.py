#!/usr/bin/env python3
"""
Setup script for Multi-Agent AI System
This script helps install all dependencies including Ollama and LLM models
"""

import os
import sys
import subprocess
import platform
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_python_dependencies():
    """Install Python dependencies from requirements.txt"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        sys.exit(1)

def check_ollama_installation():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama based on the operating system"""
    system = platform.system().lower()
    
    print(f"🔧 Installing Ollama for {system}...")
    
    if system == "windows":
        print("📥 Please download and install Ollama from: https://ollama.ai")
        print("   Or run: winget install Ollama.Ollama")
        input("   Press Enter after installing Ollama...")
        
    elif system == "darwin":  # macOS
        try:
            # Try Homebrew first
            subprocess.run(["brew", "install", "ollama"], check=True)
            print("✅ Ollama installed via Homebrew")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📥 Please download and install Ollama from: https://ollama.ai")
            input("   Press Enter after installing Ollama...")
            
    elif system == "linux":
        try:
            # Install via curl
            subprocess.run([
                "curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"
            ], shell=True, check=True)
            print("✅ Ollama installed via install script")
        except subprocess.CalledProcessError:
            print("📥 Please download and install Ollama from: https://ollama.ai")
            input("   Press Enter after installing Ollama...")
    
    else:
        print("📥 Please download and install Ollama from: https://ollama.ai")
        input("   Press Enter after installing Ollama...")

def download_models():
    """Download required LLM models"""
    models = [
        "codellama:7b-instruct",  # Primary model for code generation
        "mistral",                 # General purpose model
        "llama2"                  # Alternative model
    ]
    
    print("🤖 Downloading LLM models...")
    print("   This may take several minutes depending on your internet connection...")
    
    for model in models:
        print(f"📥 Downloading {model}...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"✅ {model} downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download {model}: {e}")
            print(f"   You can download it manually later with: ollama pull {model}")

def start_ollama_service():
    """Start Ollama service"""
    print("🚀 Starting Ollama service...")
    try:
        # Check if Ollama is already running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama service is running")
            return True
        else:
            print("⚠️ Ollama service not running. Please start it manually:")
            print("   ollama serve")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama service: {e}")
        return False

def create_startup_script():
    """Create startup scripts for easy launching"""
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Create Windows batch file
    if platform.system().lower() == "windows":
        batch_content = """@echo off
echo Starting Multi-Agent AI System...
echo.
echo 1. Starting Ollama service...
start /B ollama serve
timeout /t 5 /nobreak > nul
echo.
echo 2. Starting Backend server...
cd backend-ai
python main.py
"""
        with open(scripts_dir / "start_system.bat", "w") as f:
            f.write(batch_content)
        print("✅ Created start_system.bat")
    
    # Create shell script for Unix systems
    shell_content = """#!/bin/bash
echo "Starting Multi-Agent AI System..."
echo
echo "1. Starting Ollama service..."
ollama serve &
sleep 5
echo
echo "2. Starting Backend server..."
cd backend-ai
python main.py
"""
    with open(scripts_dir / "start_system.sh", "w") as f:
        f.write(shell_content)
    
    # Make shell script executable
    if platform.system().lower() != "windows":
        os.chmod(scripts_dir / "start_system.sh", 0o755)
    
    print("✅ Created start_system.sh")

def main():
    """Main setup function"""
    print("🚀 Multi-Agent AI System Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Check/Install Ollama
    if not check_ollama_installation():
        install_ollama()
        if not check_ollama_installation():
            print("❌ Ollama installation failed. Please install manually from https://ollama.ai")
            sys.exit(1)
    
    # Start Ollama service
    start_ollama_service()
    
    # Download models
    download_models()
    
    # Create startup scripts
    create_startup_script()
    
    print("\n" + "=" * 40)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the backend: cd backend-ai && python main.py")
    print("2. Start the frontend: cd offline-ai-frontend && npm run dev")
    print("3. Or use the startup script: ./scripts/start_system.sh")
    print("\n🌐 Backend will be available at: http://localhost:8000")
    print("🎨 Frontend will be available at: http://localhost:5173")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 