import requests
import json

# Test the exact code that's failing
code = """def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def multiply(a, b):
    return a * b"""

data = {'prompt': code}

try:
    r = requests.post('http://localhost:8000/generate-test', json=data)
    print(f'Status: {r.status_code}')
    if r.status_code == 200:
        print('SUCCESS!')
        response = r.json()
        print(f'Generated file: {response.get("file", "N/A")}')
        print('Generated test code:')
        print(response.get('code', 'No code returned')[:500])
    else:
        print('FAILED!')
        print(f'Error: {r.text}')
except Exception as e:
    print(f'Request failed: {e}') 