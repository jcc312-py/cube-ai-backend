import requests
import json
import re

def debug_test_generation():
    # Test with the same code that's failing
    code = """def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b"""
    
    print("Input code:")
    print(code)
    print("\n" + "="*50 + "\n")
    
    # Call the API and get detailed response
    try:
        response = requests.post('http://localhost:8000/generate-test', json={'prompt': code})
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response text: {response.text}")
        
        if response.status_code != 200:
            print(f"Error details: {response.text}")
            
    except Exception as e:
        print(f"Request failed: {e}")

def validate_test_code_debug(test_code: str) -> bool:
    """Debug version of validate_test_code"""
    print(f"Validating test code:")
    print(f"Test code length: {len(test_code)}")
    print(f"Test code preview: {test_code[:200]}...")
    
    pattern = r"def\s+test_\w+"
    matches = re.findall(pattern, test_code)
    print(f"Found test methods: {matches}")
    
    result = bool(re.search(pattern, test_code))
    print(f"Validation result: {result}")
    return result

if __name__ == "__main__":
    debug_test_generation() 