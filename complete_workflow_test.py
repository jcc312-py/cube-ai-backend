import requests
import json
import os
from pathlib import Path

def test_complete_workflow():
    """Test the complete AI agent workflow to answer all user questions"""
    
    print("🚀 Testing Complete AI Agent Workflow")
    print("="*60)
    
    base_url = "http://localhost:8000"
    
    # Check if server is running
    try:
        status_response = requests.get(f"{base_url}/coordination-status")
        print(f"✅ Server is running (Status: {status_response.status_code})")
    except:
        print("❌ Server is not running! Please start the backend first.")
        return
    
    # Test prompt
    prompt = "Create a function that calculates the factorial of a number with error handling"
    
    print(f"\n📝 Testing with prompt: '{prompt}'")
    print("-" * 60)
    
    # 1. Test Coordinator → Coder
    print("\n1️⃣ COORDINATOR → CODER: Code Generation")
    code_data = {
        "prompt": prompt,
        "code_history": [],
        "error_history": []
    }
    
    try:
        code_response = requests.post(f"{base_url}/generate-code", json=code_data)
        print(f"   Status: {code_response.status_code}")
        
        if code_response.status_code == 200:
            code_result = code_response.json()
            generated_code = code_result['code']
            code_file = code_result['file']
            
            print(f"   ✅ Code generated successfully!")
            print(f"   📁 File saved: {code_file}")
            print(f"   📄 Code preview: {generated_code[:100]}...")
            
            # Check if file actually exists
            file_path = Path("generated") / code_file
            if file_path.exists():
                print(f"   ✅ File confirmed saved on disk!")
            else:
                print(f"   ❌ File NOT found on disk!")
        else:
            print(f"   ❌ Code generation failed: {code_response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Code generation request failed: {e}")
        return
    
    # 2. Test Coder → Tester 
    print("\n2️⃣ CODER → TESTER: Test Generation")
    
    # First, let's check what the tester agent actually generates
    from main import tester_agent, _extract_code, validate_test_code
    
    test_prompt = f"""Generate comprehensive unit tests for the following Python code using the unittest framework.

Code to test:
```python
{generated_code}
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

    print("   🤖 Asking tester agent to generate tests...")
    raw_ai_response = tester_agent.run(test_prompt)
    print(f"   📝 Raw AI response: {raw_ai_response[:200]}...")
    
    extracted_tests = _extract_code(raw_ai_response)
    print(f"   🔍 Extracted test code: {extracted_tests[:200]}...")
    
    is_valid = validate_test_code(extracted_tests)
    print(f"   ✅ Validation result: {is_valid}")
    
    if not is_valid:
        print("   ⚠️ Let's see what the validation is looking for...")
        import re
        test_methods = re.findall(r"def\s+test_\w+", extracted_tests)
        print(f"   🔍 Found test methods: {test_methods}")
        
        if not test_methods:
            print("   ⚠️ No test methods found! The AI didn't generate proper tests.")
            print("   📋 Full generated code:")
            print(extracted_tests)
    
    # 3. Test the API endpoint
    test_data = {
        "prompt": generated_code,
        "code_history": [],
        "error_history": []
    }
    
    test_response = requests.post(f"{base_url}/generate-test", json=test_data)
    print(f"   Status: {test_response.status_code}")
    
    if test_response.status_code == 200:
        test_result = test_response.json()
        generated_tests = test_result['code']
        test_file = test_result['file']
        
        print(f"   ✅ Tests generated successfully!")
        print(f"   📁 File saved: {test_file}")
        print(f"   📄 Test preview: {generated_tests[:200]}...")
        
        # Check if test file exists
        test_file_path = Path("generated") / test_file
        if test_file_path.exists():
            print(f"   ✅ Test file confirmed saved on disk!")
        else:
            print(f"   ❌ Test file NOT found on disk!")
    else:
        print(f"   ❌ Test generation failed: {test_response.text}")
        generated_tests = None
        
    # 4. Test Tester → Runner: Test Execution
    if generated_tests:
        print("\n3️⃣ TESTER → RUNNER: Test Execution")
        
        run_data = {
            "code": generated_code,
            "test_code": generated_tests,
            "previous_errors": []
        }
        
        try:
            run_response = requests.post(f"{base_url}/run-test", json=run_data)
            print(f"   Status: {run_response.status_code}")
            
            if run_response.status_code == 200:
                run_result = run_response.json()
                test_output = run_result['output']
                coordinator_status = run_result.get('coordinator_status', 'No coordinator response')
                
                print(f"   ✅ Tests executed successfully!")
                print(f"   📊 Test Results:")
                print(f"   {test_output[:300]}...")
                print(f"   🤖 Coordinator Analysis: {coordinator_status[:200]}...")
            else:
                print(f"   ❌ Test execution failed: {run_response.text}")
                
        except Exception as e:
            print(f"   ❌ Test execution request failed: {e}")
    
    # 5. Check file generation
    print("\n4️⃣ FILE MANAGEMENT CHECK")
    
    try:
        files_response = requests.get(f"{base_url}/list-files")
        if files_response.status_code == 200:
            files = files_response.json()['files']
            print(f"   📁 Files in generated directory: {len(files)}")
            for file in files[-5:]:  # Show last 5 files
                print(f"      - {file}")
        else:
            print(f"   ❌ Failed to list files: {files_response.text}")
    except Exception as e:
        print(f"   ❌ File listing failed: {e}")
    
    # 6. Check coordination status
    print("\n5️⃣ COORDINATION STATUS")
    
    try:
        coord_response = requests.get(f"{base_url}/coordination-status")
        if coord_response.status_code == 200:
            coord_data = coord_response.json()
            print(f"   🎯 Workflow Status: {coord_data['workflow_status']}")
            print(f"   👥 Active Agents: {coord_data['active_agents']}")
            print(f"   📊 Current Phase: {coord_data['current_phase']}")
        else:
            print(f"   ❌ Failed to get coordination status: {coord_response.text}")
    except Exception as e:
        print(f"   ❌ Coordination status check failed: {e}")
    
    print("\n" + "="*60)
    print("🏁 WORKFLOW TEST COMPLETED")

if __name__ == "__main__":
    test_complete_workflow() 