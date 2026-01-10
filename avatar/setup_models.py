#!/usr/bin/env python3
"""
Avatar Model Setup Script
==========================
Downloads and configures all required models for the Avatar API.

Models downloaded:
1. InsightFace buffalo_l - Face detection and analysis
2. inswapper_128.onnx - Face swapping model
3. GFPGAN - Face enhancement (optional)

Usage:
    python setup_models.py           # Download all models
    python setup_models.py --check   # Check if models exist
    python setup_models.py --no-gfpgan  # Skip GFPGAN

Author: Dawnena Key / SonZo AI
"""

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional
import zipfile

# =============================================================================
# Configuration
# =============================================================================

# Model URLs and checksums
MODELS = {
    "inswapper_128": {
        "url": "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        "filename": "inswapper_128.onnx",
        "size_mb": 530,
        "md5": None,  # Optional verification
        "description": "Face swap model (530MB)"
    },
    "buffalo_l": {
        # buffalo_l is auto-downloaded by InsightFace to ~/.insightface/models/
        "auto_download": True,
        "description": "Face analysis model (auto-downloaded by InsightFace)"
    },
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "size_mb": 348,
        "description": "Face enhancement model (348MB)"
    }
}

# Directories
AVATAR_DIR = Path(__file__).parent.absolute()
MODELS_DIR = AVATAR_DIR / "models"
INSIGHTFACE_DIR = Path.home() / ".insightface" / "models"


# =============================================================================
# Download Utilities
# =============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    print(f"\n{desc}...")
    print(f"URL: {url}")
    print(f"Destination: {dest}")

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)
            print(f'\r  [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    if not expected_md5:
        return True

    print(f"  Verifying checksum...")
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)

    actual = md5.hexdigest()
    if actual == expected_md5:
        print(f"  ✓ Checksum verified")
        return True
    else:
        print(f"  ✗ Checksum mismatch: {actual} != {expected_md5}")
        return False


# =============================================================================
# Model Setup Functions
# =============================================================================

def setup_inswapper() -> bool:
    """Download inswapper face swap model."""
    print("\n" + "=" * 60)
    print("Setting up inswapper_128 (Face Swap Model)")
    print("=" * 60)

    model_info = MODELS["inswapper_128"]
    dest_path = MODELS_DIR / model_info["filename"]

    # Check if already exists
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        if size_mb > 500:  # Should be ~530MB
            print(f"  ✓ Model already exists: {dest_path}")
            print(f"    Size: {size_mb:.1f} MB")
            return True
        else:
            print(f"  Model exists but seems incomplete ({size_mb:.1f} MB), re-downloading...")
            dest_path.unlink()

    # Download
    success = download_with_progress(
        model_info["url"],
        dest_path,
        f"Downloading {model_info['description']}"
    )

    if success:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded successfully ({size_mb:.1f} MB)")

        # Create symlink in common location
        common_path = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"
        common_path.parent.mkdir(parents=True, exist_ok=True)
        if not common_path.exists():
            try:
                common_path.symlink_to(dest_path)
                print(f"  ✓ Created symlink: {common_path}")
            except Exception:
                # Copy instead if symlink fails
                shutil.copy2(dest_path, common_path)
                print(f"  ✓ Copied to: {common_path}")

    return success


def setup_buffalo_l() -> bool:
    """Set up InsightFace buffalo_l model (auto-downloaded)."""
    print("\n" + "=" * 60)
    print("Setting up buffalo_l (Face Analysis Model)")
    print("=" * 60)

    # Check if already exists
    buffalo_dir = INSIGHTFACE_DIR / "buffalo_l"
    if buffalo_dir.exists() and any(buffalo_dir.glob("*.onnx")):
        print(f"  ✓ Model already exists: {buffalo_dir}")
        onnx_files = list(buffalo_dir.glob("*.onnx"))
        print(f"    Found {len(onnx_files)} ONNX files")
        return True

    print("  Model will be auto-downloaded on first use by InsightFace")
    print("  Triggering download now...")

    try:
        import insightface
        from insightface.app import FaceAnalysis

        # This will trigger the download
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        print(f"  ✓ buffalo_l downloaded to: {buffalo_dir}")
        return True

    except ImportError:
        print("  ! InsightFace not installed. Install with:")
        print("    pip install insightface onnxruntime")
        return False
    except Exception as e:
        print(f"  ✗ Error downloading buffalo_l: {e}")
        return False


def setup_gfpgan() -> bool:
    """Download GFPGAN face enhancement model."""
    print("\n" + "=" * 60)
    print("Setting up GFPGAN (Face Enhancement Model)")
    print("=" * 60)

    model_info = MODELS["gfpgan"]

    # GFPGAN looks for models in specific locations
    gfpgan_dir = Path.home() / ".cache" / "gfpgan" / "weights"
    dest_path = gfpgan_dir / model_info["filename"]

    # Also copy to our models dir
    local_path = MODELS_DIR / model_info["filename"]

    # Check if already exists
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        if size_mb > 300:
            print(f"  ✓ Model already exists: {dest_path}")
            return True

    # Download
    success = download_with_progress(
        model_info["url"],
        dest_path,
        f"Downloading {model_info['description']}"
    )

    if success:
        # Copy to local models dir
        shutil.copy2(dest_path, local_path)
        print(f"  ✓ Also copied to: {local_path}")

    return success


# =============================================================================
# Verification
# =============================================================================

def check_models() -> dict:
    """Check if all required models are present."""
    print("\n" + "=" * 60)
    print("Model Status Check")
    print("=" * 60)

    status = {}

    # Check inswapper
    inswapper_path = MODELS_DIR / "inswapper_128.onnx"
    inswapper_alt = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"

    if inswapper_path.exists() or inswapper_alt.exists():
        path = inswapper_path if inswapper_path.exists() else inswapper_alt
        size_mb = path.stat().st_size / (1024 * 1024)
        status["inswapper_128"] = {"exists": True, "path": str(path), "size_mb": size_mb}
        print(f"  ✓ inswapper_128: {path} ({size_mb:.1f} MB)")
    else:
        status["inswapper_128"] = {"exists": False}
        print(f"  ✗ inswapper_128: NOT FOUND")

    # Check buffalo_l
    buffalo_dir = INSIGHTFACE_DIR / "buffalo_l"
    if buffalo_dir.exists() and any(buffalo_dir.glob("*.onnx")):
        onnx_files = list(buffalo_dir.glob("*.onnx"))
        status["buffalo_l"] = {"exists": True, "path": str(buffalo_dir), "files": len(onnx_files)}
        print(f"  ✓ buffalo_l: {buffalo_dir} ({len(onnx_files)} files)")
    else:
        status["buffalo_l"] = {"exists": False}
        print(f"  ✗ buffalo_l: NOT FOUND")

    # Check GFPGAN
    gfpgan_path = Path.home() / ".cache" / "gfpgan" / "weights" / "GFPGANv1.4.pth"
    gfpgan_alt = MODELS_DIR / "GFPGANv1.4.pth"

    if gfpgan_path.exists() or gfpgan_alt.exists():
        path = gfpgan_path if gfpgan_path.exists() else gfpgan_alt
        size_mb = path.stat().st_size / (1024 * 1024)
        status["gfpgan"] = {"exists": True, "path": str(path), "size_mb": size_mb}
        print(f"  ✓ GFPGAN: {path} ({size_mb:.1f} MB)")
    else:
        status["gfpgan"] = {"exists": False}
        print(f"  ? GFPGAN: NOT FOUND (optional)")

    # Summary
    required_ok = status["inswapper_128"]["exists"] and status["buffalo_l"]["exists"]
    print()
    if required_ok:
        print("✓ All required models are installed!")
        if not status.get("gfpgan", {}).get("exists"):
            print("  (GFPGAN is optional - face enhancement will be disabled)")
    else:
        print("✗ Some required models are missing. Run: python setup_models.py")

    return status


def test_models() -> bool:
    """Test that models can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)

    all_ok = True

    # Test InsightFace
    print("\n  Testing InsightFace...")
    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  ✓ InsightFace buffalo_l loaded successfully")
    except Exception as e:
        print(f"  ✗ InsightFace error: {e}")
        all_ok = False

    # Test inswapper
    print("\n  Testing inswapper model...")
    try:
        import onnxruntime as ort

        model_path = MODELS_DIR / "inswapper_128.onnx"
        if not model_path.exists():
            model_path = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"

        if model_path.exists():
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            print(f"  ✓ inswapper loaded: {len(session.get_inputs())} inputs, {len(session.get_outputs())} outputs")
        else:
            print("  ✗ inswapper model not found")
            all_ok = False
    except Exception as e:
        print(f"  ✗ inswapper error: {e}")
        all_ok = False

    # Test GFPGAN (optional)
    print("\n  Testing GFPGAN (optional)...")
    try:
        from gfpgan import GFPGANer

        gfpgan_path = Path.home() / ".cache" / "gfpgan" / "weights" / "GFPGANv1.4.pth"
        if not gfpgan_path.exists():
            gfpgan_path = MODELS_DIR / "GFPGANv1.4.pth"

        if gfpgan_path.exists():
            # Just check path exists, don't fully load (slow)
            print(f"  ✓ GFPGAN model found: {gfpgan_path}")
        else:
            print("  ? GFPGAN model not found (optional)")
    except ImportError:
        print("  ? GFPGAN not installed (optional)")
    except Exception as e:
        print(f"  ? GFPGAN check: {e}")

    print()
    return all_ok


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Setup Avatar API models")
    parser.add_argument("--check", action="store_true", help="Check if models exist")
    parser.add_argument("--test", action="store_true", help="Test model loading")
    parser.add_argument("--no-gfpgan", action="store_true", help="Skip GFPGAN download")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    print("=" * 60)
    print("SonZo AI - Avatar Model Setup")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    print(f"InsightFace directory: {INSIGHTFACE_DIR}")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_models()
        return

    if args.test:
        check_models()
        test_models()
        return

    # Download models
    results = {}

    # 1. inswapper (required)
    results["inswapper"] = setup_inswapper()

    # 2. buffalo_l (required)
    results["buffalo_l"] = setup_buffalo_l()

    # 3. GFPGAN (optional)
    if not args.no_gfpgan:
        results["gfpgan"] = setup_gfpgan()

    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete")
    print("=" * 60)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    # Verify
    print()
    check_models()

    # Test
    print()
    if test_models():
        print("\n✓ All models loaded successfully!")
        print("\nAvatar API is ready to use. Start with:")
        print("  python launch.py --demo")
    else:
        print("\n⚠ Some models failed to load. Check errors above.")


if __name__ == "__main__":
    main()
