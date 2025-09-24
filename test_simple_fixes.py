#!/usr/bin/env python3
"""
Simple test for basic indentation fixes
"""

def test_simple_fixes():
    """Test simple indentation fixes"""
    print("🧪 Testing simple indentation fixes...")
    
    # Test the malformed code from the user
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
    print("\n" + "="*50)
    
    # Apply simple fixing logic
    lines = malformed_code.split('\n')
    fixed_lines = []
    in_function = False
    in_docstring = False
    docstring_content = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append('')
            continue
        
        # Check if we're entering a function
        if stripped.startswith('def '):
            in_function = True
            in_docstring = False
            docstring_content = []
            fixed_lines.append(stripped)
            continue
        
        # Check if we're entering a docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_function and not in_docstring:
                in_docstring = True
                # Add proper indentation for docstring start
                fixed_lines.append('    ' + stripped)
                continue
            elif in_docstring:
                in_docstring = False
                # Add proper indentation for docstring end
                fixed_lines.append('    ' + stripped)
                continue
        
        # Handle docstring content
        if in_docstring:
            # Collect docstring content
            docstring_content.append(stripped)
            continue
        
        # If we were in a docstring but didn't find closing quotes, add them
        if docstring_content and not in_docstring:
            # Add the collected docstring content with proper indentation
            for content_line in docstring_content:
                fixed_lines.append('    ' + content_line)
            # Add closing quotes
            fixed_lines.append('    """')
            docstring_content = []
        
        # Handle function body - simple 4-space indentation
        if in_function and not in_docstring and not docstring_content:
            # Simple approach: indent everything in function body with 4 spaces
            fixed_lines.append('    ' + stripped)
            continue
        
        # Handle non-function code
        fixed_lines.append(stripped)
    
    # If we still have docstring content at the end, add closing quotes
    if docstring_content:
        for content_line in docstring_content:
            fixed_lines.append('    ' + content_line)
        fixed_lines.append('    """')
    
    fixed_code = '\n'.join(fixed_lines)
    
    print("Fixed code:")
    print(fixed_code)
    print("\n" + "="*50)
    
    # Test if the fixed code is valid Python
    try:
        compile(fixed_code, '<string>', 'exec')
        print("✅ Fixed code compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Fixed code still has syntax errors: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting simple indentation fix test...")
    
    success = test_simple_fixes()
    
    if success:
        print("\n🎉 Simple indentation fix test passed!")
        print("💡 The basic fixing logic works for the user's specific case!")
    else:
        print("\n❌ Simple indentation fix test failed!")
        print("🔧 Need to improve the fixing logic further!")

if __name__ == "__main__":
    main() 