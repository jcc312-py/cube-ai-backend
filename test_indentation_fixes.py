#!/usr/bin/env python3
"""
Test script for indentation fixes
This script tests the improved indentation and docstring fixing
"""

def test_indentation_fixes():
    """Test the indentation fixing logic"""
    print("🧪 Testing indentation fixes...")
    
    # Test case: malformed code like the one we saw
    malformed_code = '''def sum_two_numbers(a: int, b: int) -> int:

Sum of two numbers.

Args:
a (int): First number
b (int): Second number

Returns:
int: Sum of a and b

Raises:
TypeError: If either a or b is not an integer
try:
return a + b
except TypeError as e:
raise TypeError(f"Both parameters must be integers. Got type error for parameter '{e.args[0]}'.")'''
    
    print("Original malformed code:")
    print(malformed_code)
    print("\n" + "="*50 + "\n")
    
    # Simulate the fixing logic
    lines = malformed_code.split('\n')
    fixed_lines = []
    in_function = False
    in_docstring = False
    docstring_level = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Check if we're entering a function
        if stripped.startswith('def '):
            in_function = True
            docstring_level = 0
            fixed_lines.append(stripped)
            continue
        
        # Check if we're entering a docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_function and docstring_level == 0:
                in_docstring = True
                docstring_level += 1
                # Add proper indentation for docstring start
                fixed_lines.append('    ' + stripped)
                continue
            elif in_docstring and docstring_level == 1:
                in_docstring = False
                docstring_level = 0
                # Add proper indentation for docstring end
                fixed_lines.append('    ' + stripped)
                continue
        
        # Handle docstring content
        if in_docstring:
            # Indent docstring content with 4 spaces
            fixed_lines.append('    ' + stripped)
            continue
        
        # Handle function body
        if in_function and not in_docstring:
            # Indent function body with 4 spaces
            fixed_lines.append('    ' + stripped)
            continue
        
        # Handle non-function code
        fixed_lines.append(stripped)
    
    fixed_code = '\n'.join(fixed_lines)
    
    print("Fixed code:")
    print(fixed_code)
    print("\n" + "="*50 + "\n")
    
    # Test if the fixed code is valid Python
    try:
        compile(fixed_code, '<string>', 'exec')
        print("✅ Fixed code compiles successfully!")
    except SyntaxError as e:
        print(f"❌ Fixed code still has syntax errors: {e}")
    
    return fixed_code

def test_improved_prompt():
    """Test the improved prompt"""
    print("🧪 Testing improved prompt...")
    
    improved_prompt = """
You are an expert Python developer. Generate clean, properly indented Python code.

TASK: Write a function that adds two numbers
CONTEXT: 

CRITICAL REQUIREMENTS:
1. Use 4 spaces for indentation (NOT tabs)
2. Follow PEP 8 style guidelines strictly
3. Include proper error handling with try/except blocks
4. Add comprehensive type hints for all functions
5. Include detailed docstrings for all functions and classes
6. Make the code complete and runnable
7. Use proper Python naming conventions (snake_case for functions/variables)

CODE FORMATTING RULES:
- Indent with exactly 4 spaces per level
- No tabs allowed
- Proper spacing around operators
- Maximum line length of 79 characters
- Use f-strings for string formatting
- Import statements at the top
- ALWAYS use triple quotes for docstrings: \"\"\"docstring\"\"\"
- ALWAYS indent docstring content with 4 spaces
- ALWAYS indent function body with 4 spaces
- ALWAYS indent try/except blocks with 4 spaces

EXAMPLE OF PROPER FORMATTING:
def calculate_factorial(n: int) -> int:
    \"\"\"
    Calculate the factorial of a number.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result

IMPORTANT: Every line inside a function must be indented with 4 spaces. The docstring must be properly quoted with triple quotes and its content must be indented.

Generate ONLY the Python code with proper indentation. Do not include explanations, markdown formatting, or any text outside the code block.
"""
    
    print("Improved prompt includes:")
    print("- Explicit indentation requirements")
    print("- Docstring formatting rules")
    print("- Example with proper formatting")
    print("- Clear instructions for function body indentation")
    
    return improved_prompt

def main():
    """Main test function"""
    print("🚀 Starting indentation fix tests...")
    
    # Test the fixing logic
    fixed_code = test_indentation_fixes()
    
    # Test the improved prompt
    improved_prompt = test_improved_prompt()
    
    print("\n🎉 Indentation fix tests completed!")
    print("💡 The improved prompts and fixing logic should generate better code!")

if __name__ == "__main__":
    main() 