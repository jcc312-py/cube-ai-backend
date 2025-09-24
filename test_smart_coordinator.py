import requests
import json

def test_smart_coordinator():
    """Test the Smart Coordinator with different scenarios"""
    
    base_url = "http://localhost:8000"
    
    # Test scenarios
    scenarios = [
        "create a simple todo list manager",
        "build a web API for user authentication", 
        "create a basic calculator with GUI",
        "develop a data analysis script for CSV files",
        "make a simple text-based game"
    ]
    
    print("🤖 TESTING SMART COORDINATOR")
    print("="*50)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Testing: '{scenario}'")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/smart-coordinate",
                json={"prompt": scenario},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: SUCCESS")
                print(f"📋 Project Type: {data.get('plan', {}).get('project_type', 'Unknown')}")
                print(f"🎯 Goal: {data.get('plan', {}).get('main_goal', 'Unknown')}")
                print(f"📝 Code Generated: {len(data.get('generated_code', '') or '')} chars")
                print(f"🧪 Tests Generated: {len(data.get('generated_tests', '') or '')} chars")
                print(f"✨ Test Results: {'PASSED' if data.get('success') else 'FAILED'}")
                print(f"📁 Files Created: {', '.join(data.get('files_created', []))}")
                
                # Show communications
                print("\n💬 Agent Communications:")
                for comm in data.get('communications', [])[:3]:  # Show first 3
                    print(f"   • {comm}")
                
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"Error: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
        
        print()
    
    print("🏁 Smart Coordinator testing complete!")

if __name__ == "__main__":
    test_smart_coordinator() 