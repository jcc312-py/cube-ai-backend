#!/usr/bin/env python3
"""
Test script for improved prompt engineering
This script tests the enhanced prompts for better code indentation
"""

import requests
import json

def test_improved_prompts():
    """Test the improved prompt engineering via API"""
    print("🧪 Testing improved prompt engineering...")
    
    # Test cases
    test_cases = [
        {
            "name": "Simple function with proper indentation",
            "prompt": "Write a Python function that calculates the sum of all even numbers in a list"
        },
        {
            "name": "Class with methods and proper indentation", 
            "prompt": "Create a Python class called Calculator with methods for add, subtract, multiply, and divide operations"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Prompt: {test_case['prompt']}")
        
        try:
            # Send request to the API
            response = requests.post(
                "http://localhost:8000/chat",
                json={
                    "prompt": test_case['prompt'],
                    "code_history": [],
                    "error_history": []
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API request successful")
                
                # Check if code was generated
                if result.get("code"):
                    code = result["code"]
                    print(f"📝 Generated code length: {len(code)} characters")
                    
                    # Show first few lines
                    lines = code.split('\n')
                    print("First 10 lines of generated code:")
                    for j, line in enumerate(lines[:10], 1):
                        print(f"  {j:2d}: {line}")
                    
                    # Check indentation
                    indentation_issues = []
                    for j, line in enumerate(lines, 1):
                        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                            # Check if this line should be indented
                            if j > 1 and lines[j-2].strip().endswith(':'):
                                indentation_issues.append(f"Line {j}: '{line.strip()}' should be indented")
                    
                    if indentation_issues:
                        print("⚠️ Indentation issues found:")
                        for issue in indentation_issues:
                            print(f"  - {issue}")
                    else:
                        print("✅ No obvious indentation issues found")
                else:
                    print("⚠️ No code generated in response")
                    
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing API: {e}")
    
    print("\n✅ Prompt engineering test completed!")

def main():
    """Main test function"""
    print("🚀 Starting prompt engineering tests...")
    
    try:
        test_improved_prompts()
        print("\n🎉 Prompt engineering test completed successfully!")
        print("💡 The improved prompts should generate better indented code!")
    except Exception as e:
        print(f"\n❌ Prompt engineering test failed: {e}")
        print("🔧 Please check if the server is running on localhost:8000")

if __name__ == "__main__":
    main() 