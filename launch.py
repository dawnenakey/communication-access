#!/usr/bin/env python3
"""
SonZo AI - Unified Launcher
============================
Launches all SonZo services for development or production.

Usage:
    python launch.py                    # Start all services
    python launch.py --demo             # Demo mode (simulated recognition)
    python launch.py --production       # Production mode with SSL
    python launch.py --service ui       # Start only UI service

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()


class ServiceManager:
    """Manages SonZo AI services."""

    def __init__(self, demo_mode: bool = False, production: bool = False):
        self.demo_mode = demo_mode
        self.production = production
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = True

    def get_services(self) -> List[Dict]:
        """Get service configurations."""
        services = [
            {
                "name": "ui",
                "description": "UI Backend API",
                "command": [sys.executable, "ui/api.py", "--port", "8081"],
                "port": 8081,
                "health_endpoint": "/api/health"
            },
            {
                "name": "avatar",
                "description": "Avatar Generation API",
                "command": [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}/avatar')
from avatar_api import app
import uvicorn
uvicorn.run(app, host='0.0.0.0', port=8080)
"""],
                "port": 8080,
                "health_endpoint": "/api/health"
            },
        ]

        # Add recognition service (demo or real)
        if self.demo_mode:
            services.append({
                "name": "recognition",
                "description": "Recognition API (Demo Mode)",
                "command": [sys.executable, "demo/demo_recognition_api.py", "--port", "8082"],
                "port": 8082,
                "health_endpoint": "/health"
            })
        else:
            services.append({
                "name": "recognition",
                "description": "Recognition API",
                "command": [sys.executable, "demo/realtime_demo.py", "--api-only", "--port", "8082"],
                "port": 8082,
                "health_endpoint": "/health"
            })

        return services

    def start_service(self, service: Dict) -> Optional[subprocess.Popen]:
        """Start a single service."""
        name = service["name"]
        print(f"  Starting {name}: {service['description']}...")

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["SONZO_DEMO_MODE"] = "1" if self.demo_mode else "0"
            env["AVATAR_DATA_DIR"] = str(PROJECT_ROOT / "avatar_data")
            env["VIDEO_LIBRARY"] = str(PROJECT_ROOT / "avatar" / "video_library")

            process = subprocess.Popen(
                service["command"],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            self.processes[name] = process
            print(f"  ✓ {name} started on port {service['port']}")
            return process

        except Exception as e:
            print(f"  ✗ Failed to start {name}: {e}")
            return None

    def stop_all(self):
        """Stop all services."""
        print("\nStopping services...")
        self.running = False

        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"  ✓ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  ✓ {name} killed")
            except Exception as e:
                print(f"  ✗ Error stopping {name}: {e}")

    def check_health(self, service: Dict, timeout: int = 30) -> bool:
        """Check if service is healthy."""
        import urllib.request
        import urllib.error

        url = f"http://localhost:{service['port']}{service['health_endpoint']}"
        start = time.time()

        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(url, timeout=2)
                return True
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(0.5)

        return False

    def run(self, services_to_start: Optional[List[str]] = None):
        """Run all services."""
        print("=" * 60)
        print("SonZo AI - Service Launcher")
        print("=" * 60)

        if self.demo_mode:
            print("Mode: DEMO (simulated recognition)")
        elif self.production:
            print("Mode: PRODUCTION")
        else:
            print("Mode: DEVELOPMENT")

        print()

        # Get services
        services = self.get_services()

        # Filter if specific services requested
        if services_to_start:
            services = [s for s in services if s["name"] in services_to_start]

        # Start services
        print("Starting services...")
        for service in services:
            self.start_service(service)
            time.sleep(1)  # Stagger starts

        print()

        # Wait for health checks
        print("Checking service health...")
        all_healthy = True
        for service in services:
            if service["name"] in self.processes:
                if self.check_health(service):
                    print(f"  ✓ {service['name']} is healthy")
                else:
                    print(f"  ✗ {service['name']} failed health check")
                    all_healthy = False

        print()

        if all_healthy:
            print("=" * 60)
            print("All services running!")
            print("=" * 60)
            print()
            print("Access points:")
            print(f"  UI:          http://localhost:8081")
            print(f"  Avatar API:  http://localhost:8080/docs")
            print(f"  Recognition: http://localhost:8082/docs")
            print()
            print("Press Ctrl+C to stop all services")
            print()
        else:
            print("Warning: Some services failed to start")

        # Handle shutdown
        def signal_handler(sig, frame):
            self.stop_all()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Monitor processes
        try:
            while self.running:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        print(f"Warning: {name} exited with code {process.returncode}")
                        del self.processes[name]

                if not self.processes:
                    print("All services have stopped")
                    break

                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all()


def main():
    parser = argparse.ArgumentParser(description="SonZo AI Service Launcher")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (simulated recognition)")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    parser.add_argument("--service", type=str, help="Start specific service only")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")

    args = parser.parse_args()

    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return

    # Run services
    manager = ServiceManager(demo_mode=args.demo, production=args.production)

    services = [args.service] if args.service else None
    manager.run(services)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")

    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("torch", "PyTorch"),
    ]

    optional = [
        ("insightface", "InsightFace (for avatar)"),
        ("onnxruntime", "ONNX Runtime (for avatar)"),
        ("mediapipe", "MediaPipe (for hand tracking)"),
    ]

    missing_required = []
    missing_optional = []

    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (required)")
            missing_required.append(name)

    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ? {name} (optional)")
            missing_optional.append(name)

    print()

    if missing_required:
        print(f"Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install fastapi uvicorn opencv-python numpy pillow torch")
    else:
        print("All required dependencies installed!")

    if missing_optional:
        print(f"\nOptional packages not installed: {', '.join(missing_optional)}")
        print("Some features may be limited.")


if __name__ == "__main__":
    main()
