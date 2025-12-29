import requests
import sys
import json
import base64
from datetime import datetime

class ASLAPITesterFixed:
    def __init__(self, base_url="https://handtalk-58.preview.emergentagent.com"):
        self.base_url = base_url
        self.session_token = "test_session_manual"  # Use the working session
        self.user_id = "test-user-manual"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        
        headers = {'Content-Type': 'application/json'}
        if self.session_token:
            headers['Authorization'] = f'Bearer {self.session_token}'
        
        # Remove Content-Type for file uploads
        if files:
            headers.pop('Content-Type', None)

        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, headers=headers, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                if files:
                    response = requests.put(url, data=data, files=files, headers=headers, timeout=30)
                else:
                    response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if not success:
                try:
                    error_data = response.json()
                    details += f", Response: {error_data}"
                except:
                    details += f", Response: {response.text[:200]}"
            
            self.log_test(name, success, details)
            
            try:
                return success, response.json()
            except:
                return success, response.text

        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return False, {}

    def run_comprehensive_test(self):
        """Run comprehensive API tests"""
        print("🚀 Starting Comprehensive ASL API Testing...")
        print(f"Base URL: {self.base_url}")
        
        # Test basic endpoints
        print("\n=== Basic Endpoints ===")
        self.run_test("API Root", "GET", "", 200)
        self.run_test("Health Check", "GET", "health", 200)
        self.run_test("Get Signs (Public)", "GET", "signs", 200)
        
        # Test auth endpoints
        print("\n=== Authentication ===")
        self.run_test("Get Me (Authenticated)", "GET", "auth/me", 200)
        
        # Test history endpoints
        print("\n=== History Management ===")
        self.run_test("Get History", "GET", "history", 200)
        
        # Create history entry
        history_data = {
            "input_type": "asl_to_text",
            "input_content": "Test ASL signs",
            "output_content": "Hello world test",
            "confidence": 92.5
        }
        success, response = self.run_test("Create History Entry", "POST", "history", 200, data=history_data)
        
        history_id = None
        if success and isinstance(response, dict) and 'history_id' in response:
            history_id = response['history_id']
            print(f"   Created history ID: {history_id}")
        
        # Test signs CRUD
        print("\n=== Signs Dictionary CRUD ===")
        self.run_test("Search Signs (Empty)", "GET", "signs/search/nonexistentword123", 200)
        
        # Create a test sign
        test_image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        )
        
        sign_data = {
            "word": "hello",
            "description": "A greeting sign for testing"
        }
        files = {
            "image": ("hello.png", test_image_data, "image/png")
        }
        
        success, response = self.run_test("Create Sign", "POST", "signs", 200, data=sign_data, files=files)
        
        sign_id = None
        if success and isinstance(response, dict) and 'sign_id' in response:
            sign_id = response['sign_id']
            print(f"   Created sign ID: {sign_id}")
            
            # Test getting the created sign
            self.run_test("Get Sign by ID", "GET", f"signs/{sign_id}", 200)
            
            # Test updating the sign
            update_data = {
                "word": "hello_updated",
                "description": "Updated greeting sign"
            }
            self.run_test("Update Sign", "PUT", f"signs/{sign_id}", 200, data=update_data, files=files)
            
            # Test searching for updated sign
            self.run_test("Search Updated Sign", "GET", "signs/search/hello_updated", 200)
        
        # Test text-to-ASL workflow
        print("\n=== Text-to-ASL Workflow ===")
        if sign_id:
            # Search for the sign we created
            success, response = self.run_test("Search for Hello Sign", "GET", "signs/search/hello", 200)
            
            # Create another history entry for text-to-ASL
            text_to_asl_data = {
                "input_type": "text_to_asl",
                "input_content": "hello world",
                "output_content": "hello world",
                "confidence": None
            }
            self.run_test("Create Text-to-ASL History", "POST", "history", 200, data=text_to_asl_data)
        
        # Test error cases
        print("\n=== Error Handling ===")
        self.run_test("Get Non-existent Sign", "GET", "signs/nonexistent_id", 404)
        self.run_test("Invalid Endpoint", "GET", "invalid_endpoint", 404)
        
        # Cleanup - delete created resources
        print("\n=== Cleanup ===")
        if sign_id:
            self.run_test("Delete Created Sign", "DELETE", f"signs/{sign_id}", 200)
        
        if history_id:
            self.run_test("Delete History Entry", "DELETE", f"history/{history_id}", 200)
        
        # Test logout
        self.run_test("Logout", "POST", "auth/logout", 200)
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        return self.tests_passed, self.tests_run

def main():
    tester = ASLAPITesterFixed()
    passed, total = tester.run_comprehensive_test()
    
    # Save results
    with open('/app/test_reports/backend_comprehensive_results.json', 'w') as f:
        json.dump({
            "summary": {
                "tests_run": total,
                "tests_passed": passed,
                "success_rate": (passed/total*100) if total > 0 else 0
            },
            "results": tester.test_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())