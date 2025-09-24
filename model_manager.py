#!/usr/bin/env python3
"""
Model Manager for AI Agent System
Helps manage different models and their configurations
"""

import requests
import json
import sys

# API base URL
API_BASE = "http://localhost:8000"

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting models: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return None

def get_gpu_status():
    """Get GPU status and configuration"""
    try:
        response = requests.get(f"{API_BASE}/gpu-status")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error getting GPU status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return None

def switch_model(model_name):
    """Switch to a different model"""
    try:
        response = requests.post(f"{API_BASE}/switch-model", params={"model_name": model_name})
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✅ Successfully switched to {model_name}")
                print(f"📋 Model config: {json.dumps(result.get('model_config', {}), indent=2)}")
            else:
                print(f"❌ Failed to switch model: {result.get('error')}")
        else:
            print(f"❌ Error switching model: {response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")

def configure_gpu(num_gpu=1, num_thread=8, temperature=0.3):
    """Configure GPU settings"""
    try:
        response = requests.post(
            f"{API_BASE}/configure-gpu",
            params={
                "num_gpu": num_gpu,
                "num_thread": num_thread,
                "temperature": temperature
            }
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✅ GPU configuration updated")
                print(f"📋 Config: {json.dumps(result.get('config', {}), indent=2)}")
            else:
                print(f"❌ Failed to configure GPU: {result.get('error')}")
        else:
            print(f"❌ Error configuring GPU: {response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")

def show_menu():
    """Show the main menu"""
    print("\n🤖 AI Model Manager")
    print("=" * 40)
    print("1. Show available models")
    print("2. Show GPU status")
    print("3. Switch model")
    print("4. Configure GPU settings")
    print("5. Exit")
    print("=" * 40)

def main():
    """Main function"""
    while True:
        show_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n📋 Available Models:")
            models = get_available_models()
            if models:
                print(f"Current default: {models.get('current_default')}")
                print("\nModel configurations:")
                for model, config in models.get('available_models', {}).items():
                    print(f"  • {model}: {config}")
                print("\nRecommendations:")
                for task, model in models.get('recommendations', {}).items():
                    print(f"  • {task}: {model}")
        
        elif choice == "2":
            print("\n🔧 GPU Status:")
            status = get_gpu_status()
            if status:
                print(f"CUDA Available: {status.get('cuda_available')}")
                print(f"Ollama GPU Support: {status.get('ollama_gpu_support')}")
                print(f"Current Model: {status.get('current_model')}")
                print(f"Recommendation: {status.get('recommendation')}")
        
        elif choice == "3":
            print("\n🔄 Switch Model:")
            models = get_available_models()
            if models:
                available = list(models.get('available_models', {}).keys())
                print("Available models:")
                for i, model in enumerate(available, 1):
                    print(f"  {i}. {model}")
                
                try:
                    model_choice = int(input("Enter model number: ")) - 1
                    if 0 <= model_choice < len(available):
                        switch_model(available[model_choice])
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Please enter a valid number")
        
        elif choice == "4":
            print("\n⚙️ Configure GPU Settings:")
            try:
                num_gpu = int(input("Number of GPUs (default 1): ") or "1")
                num_thread = int(input("Number of threads (default 8): ") or "8")
                temperature = float(input("Temperature (default 0.3): ") or "0.3")
                configure_gpu(num_gpu, num_thread, temperature)
            except ValueError:
                print("❌ Please enter valid numbers")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 