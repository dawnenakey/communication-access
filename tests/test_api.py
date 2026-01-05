"""
API Tests for SignSync AI Backend
=================================
Run with: pytest tests/test_api.py -v

Author: Dawnena Key / SonZo AI
"""

import pytest
import os
import base64
from datetime import datetime, timezone, timedelta

# Test configuration
BASE_URL = os.environ.get("TEST_API_URL", "http://localhost:8000")


class TestHealthEndpoints:
    """Test health and basic API endpoints"""

    def test_api_root(self, client):
        """Test the API root endpoint"""
        response = client.get("/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "SignSync AI API"

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestSignsPublicEndpoints:
    """Test public sign dictionary endpoints"""

    def test_get_all_signs(self, client):
        """Test getting all signs (public endpoint)"""
        response = client.get("/api/signs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_search_signs_no_results(self, client):
        """Test searching for non-existent sign"""
        response = client.get("/api/signs/search/nonexistentword12345")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_nonexistent_sign(self, client):
        """Test getting a sign that doesn't exist"""
        response = client.get("/api/signs/nonexistent_sign_id")
        assert response.status_code == 404


class TestAuthEndpoints:
    """Test authentication endpoints"""

    def test_get_me_without_auth(self, client):
        """Test /auth/me without authentication"""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_create_session_invalid(self, client):
        """Test session creation with invalid session_id"""
        response = client.post(
            "/api/auth/session",
            json={"session_id": "invalid_session_id"}
        )
        # Should fail with 401 or 500 depending on auth service availability
        assert response.status_code in [401, 500]

    def test_logout_without_session(self, client):
        """Test logout without active session"""
        response = client.post("/api/auth/logout")
        assert response.status_code == 200


class TestSignsAuthEndpoints:
    """Test authenticated sign dictionary endpoints"""

    def test_create_sign_without_auth(self, client):
        """Test creating sign without authentication"""
        # Create a minimal test image
        test_image = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        )

        response = client.post(
            "/api/signs",
            data={"word": "test", "description": "test sign"},
            files={"image": ("test.png", test_image, "image/png")}
        )
        assert response.status_code == 401

    def test_delete_sign_without_auth(self, client):
        """Test deleting sign without authentication"""
        response = client.delete("/api/signs/some_sign_id")
        assert response.status_code == 401


class TestHistoryEndpoints:
    """Test translation history endpoints"""

    def test_get_history_without_auth(self, client):
        """Test getting history without authentication"""
        response = client.get("/api/history")
        assert response.status_code == 401

    def test_create_history_without_auth(self, client):
        """Test creating history without authentication"""
        response = client.post(
            "/api/history",
            json={
                "input_type": "asl_to_text",
                "input_content": "test",
                "output_content": "test output",
                "confidence": 0.95
            }
        )
        assert response.status_code == 401


class TestAuthenticatedEndpoints:
    """Test endpoints with authentication"""

    def test_full_sign_crud_workflow(self, authenticated_client, test_user):
        """Test complete sign CRUD workflow"""
        client = authenticated_client

        # Create a test image
        test_image = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
        )

        # CREATE - Create a new sign
        response = client.post(
            "/api/signs",
            data={"word": "pytest_test", "description": "Sign created by pytest"},
            files={"image": ("test.png", test_image, "image/png")}
        )
        assert response.status_code == 200
        sign_data = response.json()
        assert "sign_id" in sign_data
        sign_id = sign_data["sign_id"]

        # READ - Get the created sign
        response = client.get(f"/api/signs/{sign_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "pytest_test"

        # UPDATE - Update the sign
        response = client.put(
            f"/api/signs/{sign_id}",
            data={"word": "pytest_updated", "description": "Updated by pytest"},
            files={"image": ("test.png", test_image, "image/png")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["word"] == "pytest_updated"

        # SEARCH - Search for the sign
        response = client.get("/api/signs/search/pytest_updated")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        # DELETE - Delete the sign
        response = client.delete(f"/api/signs/{sign_id}")
        assert response.status_code == 200

        # VERIFY DELETE - Sign should not exist
        response = client.get(f"/api/signs/{sign_id}")
        assert response.status_code == 404

    def test_history_workflow(self, authenticated_client, test_user):
        """Test translation history workflow"""
        client = authenticated_client

        # Get initial history count
        response = client.get("/api/history")
        assert response.status_code == 200
        initial_count = len(response.json())

        # Create history entry
        response = client.post(
            "/api/history",
            json={
                "input_type": "asl_to_text",
                "input_content": "Test ASL input",
                "output_content": "Hello world",
                "confidence": 0.92
            }
        )
        assert response.status_code == 200
        history_data = response.json()
        assert "history_id" in history_data
        history_id = history_data["history_id"]

        # Verify history count increased
        response = client.get("/api/history")
        assert response.status_code == 200
        assert len(response.json()) == initial_count + 1

        # Delete the history entry
        response = client.delete(f"/api/history/{history_id}")
        assert response.status_code == 200

        # Verify history count decreased
        response = client.get("/api/history")
        assert response.status_code == 200
        assert len(response.json()) == initial_count

    def test_clear_all_history(self, authenticated_client, test_user):
        """Test clearing all history"""
        client = authenticated_client

        # Create some history entries
        for i in range(3):
            client.post(
                "/api/history",
                json={
                    "input_type": "text_to_asl",
                    "input_content": f"Test {i}",
                    "output_content": f"Output {i}",
                    "confidence": 0.85
                }
            )

        # Clear all history
        response = client.delete("/api/history")
        assert response.status_code == 200

        # Verify history is empty
        response = client.get("/api/history")
        assert response.status_code == 200
        assert len(response.json()) == 0


# ============== FIXTURES ==============

@pytest.fixture
def client():
    """Create a test client for the API"""
    from httpx import Client
    with Client(base_url=BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
def test_user(mongodb_client):
    """Create a test user in the database"""
    from datetime import datetime, timezone, timedelta
    import uuid

    db = mongodb_client

    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    session_token = f"test_session_{uuid.uuid4().hex}"

    # Create test user
    user_doc = {
        "user_id": user_id,
        "email": f"{user_id}@test.com",
        "name": "Test User",
        "picture": None,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    db.users.insert_one(user_doc)

    # Create session
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    db.user_sessions.insert_one(session_doc)

    yield {"user_id": user_id, "session_token": session_token}

    # Cleanup
    db.users.delete_one({"user_id": user_id})
    db.user_sessions.delete_one({"session_token": session_token})
    db.signs.delete_many({"created_by": user_id})
    db.translation_history.delete_many({"user_id": user_id})


@pytest.fixture
def mongodb_client():
    """Get MongoDB client"""
    from pymongo import MongoClient

    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.environ.get("DB_NAME", "signsync_test")

    client = MongoClient(mongo_url)
    db = client[db_name]

    yield db

    client.close()


@pytest.fixture
def authenticated_client(client, test_user):
    """Create an authenticated test client"""
    from httpx import Client

    with Client(
        base_url=BASE_URL,
        timeout=30.0,
        headers={"Authorization": f"Bearer {test_user['session_token']}"}
    ) as auth_client:
        yield auth_client


# ============== CONFTEST ==============

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
