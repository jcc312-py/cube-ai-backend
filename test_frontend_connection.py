import requests
import json

# Test the exact workflow that the frontend is trying to do
def test_frontend_workflow():
    base_url = "http://localhost:8000"
    
    # Step 1: Test code generation
    print("Testing code generation...")
    code_data = {
        "prompt": "Create a function that multiplies two numbers and a function that divides them with error handling",
        "code_history": [],
        "error_history": []
    }
    
    try:
        code_response = requests.post(f"{base_url}/generate-code", json=code_data)
        print(f"Code generation status: {code_response.status_code}")
        
        if code_response.status_code == 200:
            code_result = code_response.json()
            generated_code = code_result['code']
            print("Code generation successful!")
            print(f"Generated code preview: {generated_code[:100]}...")
            
            # Step 2: Test test generation with the generated code
            print("\nTesting test generation...")
            test_data = {
                "prompt": generated_code,
                "code_history": [],
                "error_history": []
            }
            
            test_response = requests.post(f"{base_url}/generate-test", json=test_data)
            print(f"Test generation status: {test_response.status_code}")
            
            if test_response.status_code == 200:
                test_result = test_response.json()
                print("Test generation successful!")
                print(f"Generated test preview: {test_result['code'][:200]}...")
            else:
                print(f"Test generation failed: {test_response.text}")
                
        else:
            print(f"Code generation failed: {code_response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_frontend_workflow() 