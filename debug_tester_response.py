from main import tester_agent, _extract_code

# Test the tester agent directly
code = """def factorial(n):
    if n < 0:
        raise ValueError("Cannot calculate factorial of negative number")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""

test_prompt = f"""Generate comprehensive unit tests for the following Python code using the unittest framework.

Code to test:
```python
{code}
```

Requirements:
- Use Python's unittest framework
- Create a test class that inherits from unittest.TestCase  
- Test all functions/methods in the code
- Include edge cases and error conditions
- Add proper assertions
- Include setUp method if needed
- Include main block: if __name__ == '__main__': unittest.main()

Return only the test code wrapped in ```python``` blocks."""

print("TESTING TESTER AGENT DIRECTLY")
print("="*50)

raw_response = tester_agent.run(test_prompt)
print("FULL RAW RESPONSE:")
print("-" * 30)
print(raw_response)
print("-" * 30)

extracted = _extract_code(raw_response)
print(f"\nEXTRACTED CODE (length: {len(extracted)}):")
print("-" * 30)
print(extracted)
print("-" * 30)

# Check if it has test methods
import re
test_methods = re.findall(r"def\s+test_\w+", extracted)
print(f"\nFOUND TEST METHODS: {test_methods}")

if test_methods:
    print("✅ Test methods found - this should work!")
else:
    print("❌ No test methods found - there's still an issue") 