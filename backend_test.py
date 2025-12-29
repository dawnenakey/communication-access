import requests
import sys
import json
import base64
from datetime import datetime
import time

class ASLAPITester:
    def __init__(self, base_url="https://handtalk-58.preview.emergentagent.com"):
        self.base_url = base_url
        self.session_token = None
        self.user_id = None
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

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, headers=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}" if not endpoint.startswith('http') else endpoint
        
        default_headers = {'Content-Type': 'application/json'}
        if self.session_token:
            default_headers['Authorization'] = f'Bearer {self.session_token}'
        
        if headers:
            default_headers.update(headers)
        
        # Remove Content-Type for file uploads
        if files:
            default_headers.pop('Content-Type', None)

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=default_headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, headers=default_headers, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=default_headers, timeout=30)
            elif method == 'PUT':
                if files:
                    response = requests.put(url, data=data, files=files, headers=default_headers, timeout=30)
                else:
                    response = requests.put(url, json=data, headers=default_headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=default_headers, timeout=30)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}, Expected: {expected_status}"
            
            if not success:
                try:
                    error_data = response.json()
                    details += f", Response: {error_data}"
                except:
                    details += f", Response: {response.text[:200]}"
            
            self.log_test(name, success, details)
            
            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return False, {}

    def test_basic_endpoints(self):
        """Test basic non-auth endpoints"""
        print("\n=== Testing Basic Endpoints ===")
        
        # Test root API endpoint
        self.run_test("API Root", "GET", "", 200)
        
        # Test health check
        self.run_test("Health Check", "GET", "health", 200)
        
        # Test signs endpoint (should work without auth)
        self.run_test("Get Signs (No Auth)", "GET", "signs", 200)

    def test_auth_endpoints(self):
        """Test authentication endpoints"""
        print("\n=== Testing Auth Endpoints ===")
        
        # Test /auth/me without token (should fail)
        self.run_test("Get Me (No Auth)", "GET", "auth/me", 401)
        
        # Test session creation with invalid session_id
        self.run_test("Create Session (Invalid)", "POST", "auth/session", 401, 
                     data={"session_id": "invalid_session_id"})

    def create_test_user_session(self):
        """Create test user and session in MongoDB for testing"""
        print("\n=== Creating Test User & Session ===")
        
        import subprocess
        
        # Generate unique IDs
        timestamp = str(int(time.time()))
        user_id = f"test-user-{timestamp}"
        session_token = f"test_session_{timestamp}"
        
        # MongoDB command to create test user and session
        mongo_cmd = f"""
        use test_database;
        db.users.insertOne({{
            user_id: "{user_id}",
            email: "test.user.{timestamp}@example.com",
            name: "Test User {timestamp}",
            picture: "https://via.placeholder.com/150",
            created_at: new Date()
        }});
        db.user_sessions.insertOne({{
            user_id: "{user_id}",
            session_token: "{session_token}",
            expires_at: new Date(Date.now() + 7*24*60*60*1000),
            created_at: new Date()
        }});
        """
        
        try:
            # Execute MongoDB command
            result = subprocess.run(
                ["mongosh", "--eval", mongo_cmd],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.session_token = session_token
                self.user_id = user_id
                print(f"✅ Created test user: {user_id}")
                print(f"✅ Created session token: {session_token}")
                return True
            else:
                print(f"❌ Failed to create test user: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating test user: {str(e)}")
            return False

    def test_auth_protected_endpoints(self):
        """Test endpoints that require authentication"""
        print("\n=== Testing Auth-Protected Endpoints ===")
        
        if not self.session_token:
            print("❌ No session token available, skipping auth tests")
            return
        
        # Test /auth/me with valid token
        self.run_test("Get Me (With Auth)", "GET", "auth/me", 200)
        
        # Test history endpoints
        self.run_test("Get History", "GET", "history", 200)
        
        # Test creating history entry
        history_data = {
            "input_type": "asl_to_text",
            "input_content": "Test ASL signs",
            "output_content": "Hello world",
            "confidence": 85.5
        }
        success, response = self.run_test("Create History", "POST", "history", 200, data=history_data)
        
        if success and 'history_id' in response:
            history_id = response['history_id']
            
            # Test deleting specific history entry
            self.run_test("Delete History Entry", "DELETE", f"history/{history_id}", 200)
        
        # Test logout
        self.run_test("Logout", "POST", "auth/logout", 200)

    def test_signs_crud(self):
        """Test sign dictionary CRUD operations"""
        print("\n=== Testing Signs CRUD Operations ===")
        
        # Test search for non-existent word
        self.run_test("Search Signs (Not Found)", "GET", "signs/search/nonexistentword", 200)
        
        if not self.session_token:
            print("❌ No session token, skipping sign creation tests")
            return
        
        # Create a test image (1x1 pixel PNG)
        test_image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        )
        
        # Test creating a sign
        sign_data = {
            "word": "test",
            "description": "Test sign for API testing"
        }
        files = {
            "image": ("test.png", test_image_data, "image/png")
        }
        
        success, response = self.run_test("Create Sign", "POST", "signs", 200, 
                                        data=sign_data, files=files)
        
        if success and 'sign_id' in response:
            sign_id = response['sign_id']
            
            # Test getting specific sign
            self.run_test("Get Sign by ID", "GET", f"signs/{sign_id}", 200)
            
            # Test updating sign
            update_data = {
                "word": "updated_test",
                "description": "Updated test sign"
            }
            self.run_test("Update Sign", "PUT", f"signs/{sign_id}", 200, 
                         data=update_data, files=files)
            
            # Test search for the created sign
            self.run_test("Search Signs (Found)", "GET", "signs/search/updated_test", 200)
            
            # Test deleting sign
            self.run_test("Delete Sign", "DELETE", f"signs/{sign_id}", 200)
            
            # Test getting deleted sign (should fail)
            self.run_test("Get Deleted Sign", "GET", f"signs/{sign_id}", 404)

    def test_error_cases(self):
        """Test various error scenarios"""
        print("\n=== Testing Error Cases ===")
        
        # Test non-existent endpoints
        self.run_test("Non-existent Endpoint", "GET", "nonexistent", 404)
        
        # Test invalid sign ID
        self.run_test("Get Invalid Sign ID", "GET", "signs/invalid_id", 404)
        
        if self.session_token:
            # Test creating sign without image
            self.run_test("Create Sign (No Image)", "POST", "signs", 422, 
                         data={"word": "test"})
            
            # Test creating sign without word
            files = {"image": ("test.png", b"fake_image_data", "image/png")}
            self.run_test("Create Sign (No Word)", "POST", "signs", 422, 
                         data={"description": "test"}, files=files)

    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting ASL API Testing...")
        print(f"Base URL: {self.base_url}")
        
        # Test basic endpoints first
        self.test_basic_endpoints()
        
        # Test auth endpoints
        self.test_auth_endpoints()
        
        # Create test user for auth-protected tests
        if self.create_test_user_session():
            self.test_auth_protected_endpoints()
            self.test_signs_crud()
        
        # Test error cases
        self.test_error_cases()
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        return self.tests_passed == self.tests_run

def main():
    tester = ASLAPITester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/test_reports/backend_test_results.json', 'w') as f:
        json.dump({
            "summary": {
                "tests_run": tester.tests_run,
                "tests_passed": tester.tests_passed,
                "success_rate": (tester.tests_passed/tester.tests_run*100) if tester.tests_run > 0 else 0
            },
            "results": tester.test_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())