import requests

code = """def factorial(n):
    if n < 0:
        raise ValueError("Cannot calculate factorial of negative number")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""

print("Testing API endpoint directly...")
r = requests.post('http://localhost:8000/generate-test', json={'prompt': code})
print(f'Status: {r.status_code}')
print(f'Response: {r.text}') 