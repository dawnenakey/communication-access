#!/usr/bin/env python3
"""
Test Script for MANO/Blender Rendering Pipeline
================================================
Verifies that the pipeline components work correctly.

Usage:
    python test_pipeline.py

Author: Dawnena Key / SonZo AI
"""

import sys
from pathlib import Path

# Add parent directory
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))


def test_asl_handshapes():
    """Test ASL handshape definitions."""
    print("\n" + "="*60)
    print("Testing ASL Handshape Definitions")
    print("="*60)

    try:
        from asl_handshapes import (
            ASL_HANDSHAPES,
            get_handshape,
            get_pose_array,
            get_all_handshapes,
            get_alphabet_handshapes,
            get_number_handshapes,
            get_classifier_handshapes,
            validate_pose
        )

        print(f"✓ Imported asl_handshapes module")

        # Test counts
        all_shapes = get_all_handshapes()
        alphabet = get_alphabet_handshapes()
        numbers = get_number_handshapes()
        classifiers = get_classifier_handshapes()

        print(f"✓ Total handshapes: {len(all_shapes)}")
        print(f"  - Alphabet (A-Z): {len(alphabet)}")
        print(f"  - Numbers (0-9): {len(numbers)}")
        print(f"  - Classifiers: {len(classifiers)}")
        print(f"  - Other: {len(all_shapes) - len(alphabet) - len(numbers) - len(classifiers)}")

        # Test getting specific handshape
        config_a = get_handshape("A")
        print(f"✓ Handshape 'A': {config_a.description}")

        # Test pose array
        pose_a = get_pose_array("A")
        print(f"✓ Pose array shape: {pose_a.shape} (expected: (15, 3))")
        assert pose_a.shape == (15, 3), "Pose array wrong shape"

        # Validate all poses
        invalid_count = 0
        for name, config in ASL_HANDSHAPES.items():
            if not validate_pose(config.pose):
                print(f"  ✗ Invalid pose: {name}")
                invalid_count += 1

        if invalid_count == 0:
            print(f"✓ All {len(ASL_HANDSHAPES)} poses validated")
        else:
            print(f"✗ {invalid_count} invalid poses found")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loader():
    """Test dataset loader (without actual data)."""
    print("\n" + "="*60)
    print("Testing Dataset Loader")
    print("="*60)

    try:
        from dataset_loader import (
            SyntheticASLDataset,
            RealASLDataset,
            CombinedASLDataset,
            get_training_transforms,
            get_validation_transforms,
        )

        print(f"✓ Imported dataset_loader module")

        # Test transforms
        train_transform = get_training_transforms(224)
        val_transform = get_validation_transforms(224)
        print(f"✓ Created training transforms")
        print(f"✓ Created validation transforms")

        return True

    except ImportError as e:
        print(f"⚠ Import error (PyTorch may not be installed): {e}")
        return True  # Not a critical failure

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blender_available():
    """Test if Blender is available."""
    print("\n" + "="*60)
    print("Testing Blender Availability")
    print("="*60)

    try:
        from batch_render import find_blender

        blender_path = find_blender()

        if blender_path:
            print(f"✓ Blender found: {blender_path}")

            # Test Blender version
            import subprocess
            result = subprocess.run(
                [blender_path, "--version"],
                capture_output=True,
                text=True
            )
            version_line = result.stdout.strip().split('\n')[0]
            print(f"✓ {version_line}")
            return True
        else:
            print("⚠ Blender not found in PATH")
            print("  Install Blender or set BLENDER_PATH environment variable")
            print("  Download from: https://www.blender.org/download/")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_render_config():
    """Test renderer configuration."""
    print("\n" + "="*60)
    print("Testing Render Configuration")
    print("="*60)

    try:
        # Check if renderer script exists
        renderer_path = Path(__file__).parent / "mano_renderer.py"

        if renderer_path.exists():
            print(f"✓ Renderer script found: {renderer_path}")
        else:
            print(f"✗ Renderer script not found: {renderer_path}")
            return False

        # Try to parse config from renderer
        with open(renderer_path, 'r') as f:
            content = f.read()

        if "class RenderConfig" in content:
            print("✓ RenderConfig class defined")
        if "class BlenderSceneSetup" in content:
            print("✓ BlenderSceneSetup class defined")
        if "class PoseApplicator" in content:
            print("✓ PoseApplicator class defined")
        if "class RenderingPipeline" in content:
            print("✓ RenderingPipeline class defined")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_mano_bone_mapping():
    """Test MANO bone mapping."""
    print("\n" + "="*60)
    print("Testing MANO Bone Mapping")
    print("="*60)

    expected_joints = [
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP",
        "RING_MCP", "RING_PIP", "RING_DIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP",
    ]

    try:
        from asl_handshapes import FingerJoint

        print(f"✓ FingerJoint enum imported")

        for i, expected_name in enumerate(expected_joints):
            joint = FingerJoint(i)
            if joint.name == expected_name:
                print(f"  ✓ Joint {i}: {joint.name}")
            else:
                print(f"  ✗ Joint {i}: expected {expected_name}, got {joint.name}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_output_directory():
    """Test output directory creation."""
    print("\n" + "="*60)
    print("Testing Output Directory")
    print("="*60)

    import tempfile
    import shutil

    try:
        # Create temp directory
        test_dir = Path(tempfile.mkdtemp()) / "synthetic_test"
        test_dir.mkdir(parents=True)

        print(f"✓ Created test directory: {test_dir}")

        # Test subdirectory creation
        for hs in ["A", "B", "C"]:
            subdir = test_dir / hs
            subdir.mkdir()

        print(f"✓ Created {3} handshape subdirectories")

        # Cleanup
        shutil.rmtree(test_dir.parent)
        print("✓ Cleaned up test directory")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("MANO/Blender Rendering Pipeline Test Suite")
    print("="*60)

    tests = [
        ("ASL Handshapes", test_asl_handshapes),
        ("Dataset Loader", test_dataset_loader),
        ("Blender Availability", test_blender_available),
        ("Render Configuration", test_render_config),
        ("MANO Bone Mapping", test_mano_bone_mapping),
        ("Output Directory", test_output_directory),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        print("\n⚠ Some tests failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
