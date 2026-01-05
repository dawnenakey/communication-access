"""
Pytest configuration and fixtures for SignSync AI tests

Usage:
    pytest tests/ -v
    pytest tests/ -v --cov=backend
"""

import pytest
import os


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--api-url",
        action="store",
        default="http://localhost:8000",
        help="API base URL for integration tests"
    )
    parser.addoption(
        "--mongo-url",
        action="store",
        default="mongodb://localhost:27017",
        help="MongoDB connection URL"
    )
    parser.addoption(
        "--db-name",
        action="store",
        default="signsync_test",
        help="Test database name"
    )


@pytest.fixture(scope="session")
def api_url(request):
    """Get API URL from command line or environment"""
    return request.config.getoption("--api-url") or os.environ.get("TEST_API_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def mongo_url(request):
    """Get MongoDB URL from command line or environment"""
    return request.config.getoption("--mongo-url") or os.environ.get("MONGO_URL", "mongodb://localhost:27017")


@pytest.fixture(scope="session")
def db_name(request):
    """Get database name from command line or environment"""
    return request.config.getoption("--db-name") or os.environ.get("DB_NAME", "signsync_test")
