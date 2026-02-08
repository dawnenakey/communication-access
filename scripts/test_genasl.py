#!/usr/bin/env python3
"""
GenASL Test Script

This script tests the GenASL API endpoints and can generate marketing videos.

Usage:
    # Test all endpoints
    python scripts/test_genasl.py --test-all

    # Generate a video for a phrase
    python scripts/test_genasl.py --generate "Hello, how are you?"

    # Download generated video
    python scripts/test_genasl.py --download <execution_id>

    # List available signs
    python scripts/test_genasl.py --list-signs

    # Generate marketing videos
    python scripts/test_genasl.py --marketing

Environment:
    API_URL: Base URL for the API (default: http://localhost:8000)
    AUTH_TOKEN: Bearer token for authentication (optional)
"""

import argparse
import json
import os
import sys
import time
from typing import Optional
import urllib.request
import urllib.error


API_URL = os.getenv("API_URL", "http://localhost:8000")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# Marketing phrases for video generation
MARKETING_PHRASES = [
    "Hello, welcome to SonZo",
    "Thank you for watching",
    "Learn sign language with us",
    "Communication for everyone",
    "I love ASL",
    "Nice to meet you",
    "How are you today?",
    "Good morning",
    "Goodbye, see you soon",
    "Help is available",
]


def make_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[dict] = None,
    timeout: int = 60
) -> dict:
    """Make an HTTP request to the API."""
    url = f"{API_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    req_data = json.dumps(data).encode() if data else None
    request = urllib.request.Request(url, data=req_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response: {error_body}")
        raise
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        raise


def test_health():
    """Test the health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    try:
        # Test main backend health
        response = make_request("/api/health")
        print(f"Backend Health: {json.dumps(response, indent=2)}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_genasl_signs():
    """Test the GenASL signs endpoint."""
    print("\n=== Testing GenASL Signs Endpoint ===")
    try:
        response = make_request("/api/genasl/signs")
        signs = response.get("signs", [])
        print(f"Available signs: {len(signs)}")
        if signs:
            print(f"Sample signs: {signs[:10]}...")
        return True
    except Exception as e:
        print(f"GenASL signs test failed: {e}")
        return False


def test_genasl_translate(text: str = "Hello, how are you?"):
    """Test the GenASL translation endpoint."""
    print(f"\n=== Testing GenASL Translation ===")
    print(f"Input: {text}")
    try:
        response = make_request("/api/genasl/translate", method="POST", data={"text": text})
        print(f"Translation result: {json.dumps(response, indent=2)}")
        return response
    except Exception as e:
        print(f"GenASL translation test failed: {e}")
        return None


def generate_video(text: str, wait: bool = True, max_wait: int = 120):
    """Generate a GenASL video for the given text."""
    print(f"\n=== Generating GenASL Video ===")
    print(f"Input: {text}")

    try:
        response = make_request(
            "/api/genasl/generate",
            method="POST",
            data={"text": text, "avatar_style": "realistic"},
            timeout=max_wait
        )

        execution_id = response.get("execution_id")
        status = response.get("status")
        video_url = response.get("video_url")

        print(f"Execution ID: {execution_id}")
        print(f"Initial Status: {status}")

        if video_url:
            print(f"Video URL: {video_url}")
            return response

        if not wait or status == "SUCCEEDED":
            return response

        # Poll for completion
        print("Waiting for video generation...")
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = check_status(execution_id)
            status = status_response.get("status")
            print(f"  Status: {status}")

            if status == "SUCCEEDED":
                video_url = status_response.get("video_url")
                print(f"Video URL: {video_url}")
                return status_response
            elif status in ["FAILED", "TIMED_OUT", "ABORTED"]:
                print(f"Generation failed: {status}")
                return status_response

            time.sleep(5)

        print("Timeout waiting for video generation")
        return response

    except Exception as e:
        print(f"Video generation failed: {e}")
        return None


def check_status(execution_id: str):
    """Check the status of a GenASL execution."""
    print(f"\n=== Checking Execution Status ===")
    print(f"Execution ID: {execution_id}")
    try:
        response = make_request(f"/api/genasl/status/{execution_id}")
        print(f"Status: {json.dumps(response, indent=2)}")
        return response
    except Exception as e:
        print(f"Status check failed: {e}")
        return {"status": "ERROR", "error": str(e)}


def download_video(url: str, output_path: str):
    """Download a video from URL."""
    print(f"\n=== Downloading Video ===")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def generate_marketing_videos(output_dir: str = "marketing_videos"):
    """Generate marketing videos for predefined phrases."""
    print("\n=== Generating Marketing Videos ===")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, phrase in enumerate(MARKETING_PHRASES):
        print(f"\n[{i+1}/{len(MARKETING_PHRASES)}] Generating: {phrase}")
        result = generate_video(phrase, wait=True, max_wait=180)

        if result and result.get("video_url"):
            # Download the video
            safe_name = phrase.lower().replace(" ", "_").replace(",", "")[:30]
            output_path = os.path.join(output_dir, f"{i+1:02d}_{safe_name}.mp4")
            if download_video(result["video_url"], output_path):
                results.append({
                    "phrase": phrase,
                    "video_url": result["video_url"],
                    "local_path": output_path,
                    "status": "success"
                })
            else:
                results.append({
                    "phrase": phrase,
                    "video_url": result.get("video_url"),
                    "status": "download_failed"
                })
        else:
            results.append({
                "phrase": phrase,
                "status": "generation_failed"
            })

    # Summary
    print("\n=== Marketing Video Generation Summary ===")
    success = sum(1 for r in results if r["status"] == "success")
    print(f"Generated: {success}/{len(MARKETING_PHRASES)} videos")

    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['phrase']}: {r['status']}")

    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def run_all_tests():
    """Run all GenASL tests."""
    print("=" * 60)
    print("GenASL API Test Suite")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Auth Token: {'Set' if AUTH_TOKEN else 'Not set'}")

    results = {
        "health": test_health(),
        "signs": test_genasl_signs(),
        "translate": test_genasl_translate() is not None,
    }

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="GenASL API Test Script")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--health", action="store_true", help="Test health endpoint")
    parser.add_argument("--list-signs", action="store_true", help="List available signs")
    parser.add_argument("--translate", type=str, help="Translate text to ASL gloss")
    parser.add_argument("--generate", type=str, help="Generate video for text")
    parser.add_argument("--status", type=str, help="Check execution status")
    parser.add_argument("--download", type=str, help="Download video from URL")
    parser.add_argument("--output", type=str, default="video.mp4", help="Output path for download")
    parser.add_argument("--marketing", action="store_true", help="Generate marketing videos")
    parser.add_argument("--api-url", type=str, help="Override API URL")

    args = parser.parse_args()

    if args.api_url:
        global API_URL
        API_URL = args.api_url

    if args.test_all:
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if args.health:
        test_health()
        return

    if args.list_signs:
        test_genasl_signs()
        return

    if args.translate:
        test_genasl_translate(args.translate)
        return

    if args.generate:
        generate_video(args.generate)
        return

    if args.status:
        check_status(args.status)
        return

    if args.download:
        download_video(args.download, args.output)
        return

    if args.marketing:
        generate_marketing_videos()
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
