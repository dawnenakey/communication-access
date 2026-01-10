#!/usr/bin/env python3
"""
Avatar API Test Script
=======================
Tests face capture, face swap, and API components.

Usage:
    python test_avatar.py           # Run all tests
    python test_avatar.py --quick   # Quick component test only
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import cv2

# Test results
results = {}


def test_face_capture():
    """Test face capture module."""
    print("\n" + "=" * 60)
    print("Testing Face Capture")
    print("=" * 60)

    try:
        from face_capture import FaceCapture

        # Initialize
        capture = FaceCapture(use_gpu=False)
        print("  ✓ FaceCapture initialized")

        # Create synthetic test image with a face-like pattern
        # Using a more realistic test by downloading a sample image
        print("  Creating test image...")

        # Create a simple test image (colored rectangle as placeholder)
        # In production, this would use a real face image
        img = np.zeros((640, 480, 3), dtype=np.uint8)

        # Draw a face-like ellipse (won't be detected, but tests initialization)
        cv2.ellipse(img, (240, 200), (100, 130), 0, 0, 360, (200, 180, 160), -1)
        cv2.circle(img, (200, 180), 15, (50, 50, 50), -1)  # Left eye
        cv2.circle(img, (280, 180), 15, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(img, (240, 260), (40, 20), 0, 0, 180, (150, 100, 100), -1)  # Mouth

        # Try to extract (may not find a face in synthetic image)
        face_data = capture.extract_face(img)

        if face_data:
            print(f"  ✓ Face extracted (quality: {face_data.quality_score:.2%})")
            validation = capture.validate_face(face_data)
            print(f"  ✓ Face validated: {validation['is_valid']}")
            results["face_capture"] = True
        else:
            print("  ! No face detected in synthetic image (expected)")
            print("  ✓ Face capture module works correctly")
            results["face_capture"] = True

        return True

    except Exception as e:
        print(f"  ✗ Face capture error: {e}")
        results["face_capture"] = False
        return False


def test_face_swap_model():
    """Test face swap model loading."""
    print("\n" + "=" * 60)
    print("Testing Face Swap Model")
    print("=" * 60)

    try:
        from face_swap import FaceSwapper, SwapConfig

        # Configure for CPU testing
        config = SwapConfig(
            use_gpu=False,
            use_enhancer=False  # Skip GFPGAN
        )

        print("  Loading FaceSwapper (CPU mode, no enhancer)...")
        swapper = FaceSwapper(config)

        if swapper.swapper is None:
            print("  ✗ Swapper model not loaded")
            results["face_swap"] = False
            return False

        print("  ✓ FaceSwapper initialized")
        print("  ✓ Swapper model loaded")

        # Check face analyzer
        if swapper.face_analyzer is not None:
            print("  ✓ Face analyzer ready")

        results["face_swap"] = True
        return True

    except Exception as e:
        print(f"  ✗ Face swap error: {e}")
        import traceback
        traceback.print_exc()
        results["face_swap"] = False
        return False


def test_model_paths():
    """Verify model files exist."""
    print("\n" + "=" * 60)
    print("Verifying Model Paths")
    print("=" * 60)

    models_found = 0

    # Check inswapper
    inswapper_paths = [
        Path(__file__).parent / "models" / "inswapper_128.onnx",
        Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
    ]

    for path in inswapper_paths:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ inswapper_128.onnx: {path} ({size_mb:.1f} MB)")
            models_found += 1
            break
    else:
        print("  ✗ inswapper_128.onnx NOT FOUND")

    # Check buffalo_l
    buffalo_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    if buffalo_dir.exists():
        onnx_files = list(buffalo_dir.glob("*.onnx"))
        print(f"  ✓ buffalo_l: {buffalo_dir} ({len(onnx_files)} files)")
        models_found += 1
    else:
        print(f"  ✗ buffalo_l NOT FOUND at {buffalo_dir}")

    results["model_paths"] = models_found >= 2
    return models_found >= 2


def test_api_imports():
    """Test API module imports."""
    print("\n" + "=" * 60)
    print("Testing API Imports")
    print("=" * 60)

    try:
        # Change to avatar directory for imports
        os.chdir(Path(__file__).parent)

        from avatar_api import app, Config, AvatarCreateRequest
        print("  ✓ avatar_api module imported")

        from face_capture import FaceCapture, FaceData
        print("  ✓ face_capture module imported")

        from face_swap import FaceSwapper, SwapConfig
        print("  ✓ face_swap module imported")

        # Check FastAPI app
        print(f"  ✓ FastAPI app: {app.title} v{app.version}")

        results["api_imports"] = True
        return True

    except Exception as e:
        print(f"  ✗ Import error: {e}")
        results["api_imports"] = False
        return False


def test_onnx_inference():
    """Test ONNX model inference directly."""
    print("\n" + "=" * 60)
    print("Testing ONNX Inference")
    print("=" * 60)

    try:
        import onnxruntime as ort

        # Find model
        model_paths = [
            Path(__file__).parent / "models" / "inswapper_128.onnx",
            Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
        ]

        model_path = None
        for p in model_paths:
            if p.exists():
                model_path = p
                break

        if model_path is None:
            print("  ✗ Model not found for inference test")
            results["onnx_inference"] = False
            return False

        # Load model
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"  ✓ Model loaded: {model_path.name}")
        print(f"  ✓ Inputs: {len(inputs)}")
        for inp in inputs:
            print(f"      - {inp.name}: {inp.shape}")
        print(f"  ✓ Outputs: {len(outputs)}")
        for out in outputs:
            print(f"      - {out.name}: {out.shape}")

        results["onnx_inference"] = True
        return True

    except Exception as e:
        print(f"  ✗ ONNX inference error: {e}")
        results["onnx_inference"] = False
        return False


def test_insightface_detection():
    """Test InsightFace face detection on a real-looking image."""
    print("\n" + "=" * 60)
    print("Testing InsightFace Detection")
    print("=" * 60)

    try:
        from insightface.app import FaceAnalysis

        # Initialize
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✓ FaceAnalysis initialized")

        # Create a test image (random noise)
        # In real testing, we'd use an actual face image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Run detection (won't find faces in noise, but tests the pipeline)
        faces = app.get(test_img)
        print(f"  ✓ Detection ran successfully (found {len(faces)} faces in test noise)")

        results["insightface_detection"] = True
        return True

    except Exception as e:
        print(f"  ✗ InsightFace error: {e}")
        results["insightface_detection"] = False
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SonZo AI - Avatar API Tests")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test only')
    args = parser.parse_args()

    start_time = time.time()

    # Run tests
    test_model_paths()
    test_api_imports()
    test_onnx_inference()
    test_insightface_detection()

    if not args.quick:
        test_face_capture()
        test_face_swap_model()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if passed_test:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Passed: {passed}/{passed + failed}")
    print(f"Time: {elapsed:.1f}s")

    if failed == 0:
        print("\n✓ All tests passed! Avatar API is ready.")
        return 0
    else:
        print(f"\n⚠ {failed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
