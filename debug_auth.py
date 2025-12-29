import requests
import json

# Test with a simple session token
url = "https://handtalk-58.preview.emergentagent.com/api/auth/me"
headers = {"Authorization": "Bearer test_session_1767031995"}

print("Testing auth endpoint with session token...")
response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")

# Also test MongoDB connection directly
import subprocess
result = subprocess.run(
    ["mongosh", "--eval", "use test_database; db.user_sessions.find({}, {session_token: 1, user_id: 1}).limit(3)"],
    capture_output=True,
    text=True
)
print(f"\nMongoDB query result:")
print(f"Return code: {result.returncode}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")