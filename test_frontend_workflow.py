import requests
import json

def test_frontend_workflow():
    """Test the exact workflow that the frontend is doing"""
    
    print("🔍 Testing Frontend Workflow Replication")
    print("="*50)
    
    # Step 1: Generate code (like frontend does)
    print("\n1. Code Generation:")
    code_response = requests.post('http://localhost:8000/generate-code', json={
        'prompt': 'create a simple python calculator',
        'code_history': [],
        'error_history': []
    })
    
    print(f"   Status: {code_response.status_code}")
    if code_response.status_code == 200:
        code_data = code_response.json()
        generated_code = code_data['code']
        print(f"   ✅ Code generated: {len(generated_code)} chars")
        print(f"   📄 Preview: {generated_code[:100]}...")
    else:
        print(f"   ❌ Failed: {code_response.text}")
        return
    
    # Step 2: Generate tests (like frontend does)
    print("\n2. Test Generation:")
    test_response = requests.post('http://localhost:8000/generate-test', json={
        'prompt': generated_code,  # Frontend passes the generated code as prompt
        'code_history': [],
        'error_history': []
    })
    
    print(f"   Status: {test_response.status_code}")
    if test_response.status_code == 200:
        test_data = test_response.json()
        generated_tests = test_data['code']
        print(f"   ✅ Tests generated: {len(generated_tests)} chars")
        print(f"   📄 Preview: {generated_tests[:200]}...")
        
        # Check if tests contain actual test methods
        import re
        test_methods = re.findall(r"def\s+test_\w+", generated_tests)
        print(f"   🔍 Test methods found: {test_methods}")
    else:
        print(f"   ❌ Failed: {test_response.text}")
        return
    
    # Step 3: Run tests (like frontend does)
    print("\n3. Test Execution:")
    run_response = requests.post('http://localhost:8000/run-test', json={
        'code': generated_code,
        'test_code': generated_tests,
        'previous_errors': []
    })
    
    print(f"   Status: {run_response.status_code}")
    if run_response.status_code == 200:
        run_data = run_response.json()
        test_output = run_data['output']
        coordinator_status = run_data.get('coordinator_status', 'No coordinator response')
        
        print(f"   ✅ Tests executed!")
        print(f"   📊 Results: {test_output}")
        print(f"   🤖 Coordinator: {coordinator_status[:100]}...")
        
        # Check if any tests actually ran
        if "Total Tests: 0" in test_output:
            print("   ⚠️ WARNING: 0 tests executed!")
            print("   🔍 Full test code being executed:")
            print(f"   {generated_tests}")
    else:
        print(f"   ❌ Failed: {run_response.text}")
    
    print("\n" + "="*50)
    print("🏁 Workflow test complete!")

if __name__ == "__main__":
    test_frontend_workflow() 