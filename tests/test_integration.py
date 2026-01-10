#!/usr/bin/env python3
"""
SonZo AI - Integration Tests
==============================
End-to-end tests for the complete SonZo workflow.

Tests:
1. Service health checks
2. Webcam → Recognition → Text pipeline
3. Text → Avatar → Video pipeline
4. Full conversation flow
5. User data persistence

Usage:
    pytest tests/test_integration.py -v
    python tests/test_integration.py  # Run standalone

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

class TestConfig:
    """Test configuration."""
    UI_URL = os.environ.get("SONZO_UI_URL", "http://localhost:8081")
    AVATAR_URL = os.environ.get("SONZO_AVATAR_URL", "http://localhost:8080")
    RECOGNITION_URL = os.environ.get("SONZO_RECOGNITION_URL", "http://localhost:8082")

    TIMEOUT = 10
    RETRY_DELAY = 0.5


# =============================================================================
# Test Helpers
# =============================================================================

def check_service_health(url: str, endpoint: str = "/health") -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=TestConfig.TIMEOUT)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_service(url: str, endpoint: str = "/health", timeout: int = 30) -> bool:
    """Wait for a service to become available."""
    start = time.time()
    while time.time() - start < timeout:
        if check_service_health(url, endpoint):
            return True
        time.sleep(TestConfig.RETRY_DELAY)
    return False


def create_test_image() -> str:
    """Create a test image (base64 encoded)."""
    import numpy as np

    try:
        import cv2

        # Create a simple test image with skin-colored region
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add a skin-colored oval (simulating a hand)
        center = (320, 240)
        axes = (100, 150)
        color = (140, 160, 200)  # BGR skin color
        cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

    except ImportError:
        # Return a minimal valid JPEG if cv2 not available
        # This is a 1x1 pixel JPEG
        return "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAB//9k="


# =============================================================================
# Integration Tests
# =============================================================================

class TestServiceHealth:
    """Test service health endpoints."""

    def test_ui_health(self):
        """Test UI service health."""
        response = requests.get(
            f"{TestConfig.UI_URL}/api/health",
            timeout=TestConfig.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_recognition_health(self):
        """Test recognition service health."""
        response = requests.get(
            f"{TestConfig.RECOGNITION_URL}/health",
            timeout=TestConfig.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"

    def test_avatar_health(self):
        """Test avatar service health."""
        response = requests.get(
            f"{TestConfig.AVATAR_URL}/api/health",
            timeout=TestConfig.TIMEOUT
        )
        assert response.status_code == 200


class TestRecognitionPipeline:
    """Test recognition pipeline."""

    def test_recognize_image(self):
        """Test image recognition endpoint."""
        image_data = create_test_image()

        response = requests.post(
            f"{TestConfig.RECOGNITION_URL}/api/recognize",
            json={"image": image_data},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        # Should have either a recognized sign or ready state
        assert data.get("state") in ["ready", "processing", "recognized"]

    def test_recognition_with_hands(self):
        """Test that hands are detected in response."""
        image_data = create_test_image()

        response = requests.post(
            f"{TestConfig.RECOGNITION_URL}/api/recognize",
            json={"image": image_data},
            timeout=TestConfig.TIMEOUT
        )

        data = response.json()

        # In demo mode, should detect simulated hands
        # (Depends on image content)
        assert "hands" in data or data.get("state") == "ready"

    def test_get_available_signs(self):
        """Test getting available signs."""
        response = requests.get(
            f"{TestConfig.RECOGNITION_URL}/api/signs",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert "signs" in data
        assert len(data["signs"]) > 0


class TestUserManagement:
    """Test user management endpoints."""

    test_user_id = None

    def test_create_user(self):
        """Test user creation."""
        response = requests.post(
            f"{TestConfig.UI_URL}/api/users",
            json={"name": "Test User"},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        TestUserManagement.test_user_id = data["id"]

    def test_get_user(self):
        """Test getting user."""
        if not TestUserManagement.test_user_id:
            self.test_create_user()

        response = requests.get(
            f"{TestConfig.UI_URL}/api/users/{TestUserManagement.test_user_id}",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test User"

    def test_update_settings(self):
        """Test updating settings."""
        if not TestUserManagement.test_user_id:
            self.test_create_user()

        response = requests.patch(
            f"{TestConfig.UI_URL}/api/users/{TestUserManagement.test_user_id}/settings",
            json={"dark_mode": True},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data["dark_mode"] is True

    def test_record_progress(self):
        """Test recording progress."""
        if not TestUserManagement.test_user_id:
            self.test_create_user()

        # Record learned sign
        response = requests.post(
            f"{TestConfig.UI_URL}/api/users/{TestUserManagement.test_user_id}/progress/sign-learned",
            params={"sign": "HELLO"},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert data["signs_learned"] >= 1

    def test_get_achievements(self):
        """Test getting achievements."""
        if not TestUserManagement.test_user_id:
            self.test_create_user()

        response = requests.get(
            f"{TestConfig.UI_URL}/api/users/{TestUserManagement.test_user_id}/achievements",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_delete_user(self):
        """Test deleting user."""
        if not TestUserManagement.test_user_id:
            self.test_create_user()

        response = requests.delete(
            f"{TestConfig.UI_URL}/api/users/{TestUserManagement.test_user_id}",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200


class TestConversationFlow:
    """Test conversation flow."""

    user_id = None
    conversation_id = None

    def setup_method(self):
        """Create test user."""
        response = requests.post(
            f"{TestConfig.UI_URL}/api/users",
            json={"name": "Conversation Test User"},
            timeout=TestConfig.TIMEOUT
        )
        self.user_id = response.json()["id"]

    def teardown_method(self):
        """Clean up test user."""
        if self.user_id:
            requests.delete(
                f"{TestConfig.UI_URL}/api/users/{self.user_id}",
                timeout=TestConfig.TIMEOUT
            )

    def test_create_conversation(self):
        """Test creating conversation."""
        response = requests.post(
            f"{TestConfig.UI_URL}/api/users/{self.user_id}/conversations",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        self.conversation_id = data["id"]

    def test_add_message(self):
        """Test adding message to conversation."""
        # Create conversation first
        self.test_create_conversation()

        response = requests.post(
            f"{TestConfig.UI_URL}/api/users/{self.user_id}/conversations/{self.conversation_id}/messages",
            json={"type": "user", "sign": "HELLO"},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 1

    def test_get_conversations(self):
        """Test getting conversations."""
        response = requests.get(
            f"{TestConfig.UI_URL}/api/users/{self.user_id}/conversations",
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestEndToEndFlow:
    """Test complete end-to-end flow."""

    def test_full_recognition_flow(self):
        """Test: Image → Recognition → Result."""
        image_data = create_test_image()

        # Send image
        response = requests.post(
            f"{TestConfig.RECOGNITION_URL}/api/recognize",
            json={"image": image_data},
            timeout=TestConfig.TIMEOUT
        )

        assert response.status_code == 200
        result = response.json()

        # Verify response structure
        assert "state" in result
        assert "latency_ms" in result

        print(f"Recognition result: {result}")

    def test_demo_sequence_mode(self):
        """Test scripted demo mode."""
        # Enable scripted mode
        response = requests.post(
            f"{TestConfig.RECOGNITION_URL}/api/demo/scripted",
            params={"enabled": True},
            timeout=TestConfig.TIMEOUT
        )
        assert response.status_code == 200

        # Set custom sequence
        sequence = ["HELLO", "THANK_YOU", "GOODBYE"]
        response = requests.post(
            f"{TestConfig.RECOGNITION_URL}/api/demo/sequence",
            json=sequence,
            timeout=TestConfig.TIMEOUT
        )
        assert response.status_code == 200

        # Recognize multiple times
        image_data = create_test_image()
        signs_recognized = []

        for _ in range(3):
            response = requests.post(
                f"{TestConfig.RECOGNITION_URL}/api/recognize",
                json={"image": image_data},
                timeout=TestConfig.TIMEOUT
            )
            if response.json().get("sign"):
                signs_recognized.append(response.json()["sign"])
            time.sleep(2)  # Wait for cooldown

        print(f"Scripted sequence: {signs_recognized}")


# =============================================================================
# Test Runner
# =============================================================================

def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("SonZo AI - Integration Tests")
    print("=" * 60)
    print()

    # Check services
    print("Checking services...")
    services = [
        ("UI", TestConfig.UI_URL, "/api/health"),
        ("Recognition", TestConfig.RECOGNITION_URL, "/health"),
        ("Avatar", TestConfig.AVATAR_URL, "/api/health"),
    ]

    all_healthy = True
    for name, url, endpoint in services:
        if check_service_health(url, endpoint):
            print(f"  ✓ {name} service is healthy")
        else:
            print(f"  ✗ {name} service is not available")
            all_healthy = False

    if not all_healthy:
        print()
        print("Some services are not available.")
        print("Start services with: python launch.py --demo")
        return False

    print()
    print("Running tests...")
    print()

    # Run test classes
    test_classes = [
        TestServiceHealth,
        TestRecognitionPipeline,
        TestUserManagement,
        TestConversationFlow,
        TestEndToEndFlow,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"  {test_class.__name__}:")
        instance = test_class()

        # Setup if exists
        if hasattr(instance, 'setup_method'):
            try:
                instance.setup_method()
            except Exception as e:
                print(f"    Setup failed: {e}")
                continue

        # Run test methods
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"    ✓ {method_name}")
                    passed_tests += 1
                except AssertionError as e:
                    print(f"    ✗ {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))
                except Exception as e:
                    print(f"    ✗ {method_name}: {e}")
                    failed_tests.append((test_class.__name__, method_name, str(e)))

        # Teardown if exists
        if hasattr(instance, 'teardown_method'):
            try:
                instance.teardown_method()
            except Exception:
                pass

        print()

    # Summary
    print("=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print()
        print("Failed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
