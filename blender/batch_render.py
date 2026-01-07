#!/usr/bin/env python3
"""
Batch Rendering Orchestrator for ASL Synthetic Data
====================================================
Orchestrates Blender rendering across multiple processes.

Usage:
    python batch_render.py --output ./synthetic_data --samples 100 --parallel 4

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional

# Add parent directory
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from asl_handshapes import get_all_handshapes, get_alphabet_handshapes


def find_blender() -> Optional[str]:
    """Find Blender executable."""
    possible_paths = [
        # Linux
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        os.path.expanduser("~/blender/blender"),
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        # Windows
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    ]

    # Check environment variable first
    if "BLENDER_PATH" in os.environ:
        return os.environ["BLENDER_PATH"]

    # Check PATH
    import shutil
    blender_in_path = shutil.which("blender")
    if blender_in_path:
        return blender_in_path

    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def render_handshape(
    blender_path: str,
    handshape: str,
    output_dir: str,
    samples: int,
    model_path: Optional[str] = None,
    image_size: int = 512
) -> Dict:
    """
    Render a single handshape using Blender subprocess.

    Args:
        blender_path: Path to Blender executable
        handshape: Handshape name (e.g., "A")
        output_dir: Output directory
        samples: Number of samples to render
        model_path: Path to MANO model (optional)
        image_size: Output image size

    Returns:
        Dict with rendering results
    """
    script_path = Path(__file__).parent / "mano_renderer.py"

    cmd = [
        blender_path,
        "--background",
        "--python", str(script_path),
        "--",
        "--handshape", handshape,
        "--output", output_dir,
        "--samples", str(samples),
        "--size", str(image_size),
    ]

    if model_path:
        cmd.extend(["--model", model_path])

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        return {
            "handshape": handshape,
            "success": result.returncode == 0,
            "elapsed_time": elapsed,
            "stdout": result.stdout[-2000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "handshape": handshape,
            "success": False,
            "elapsed_time": 3600,
            "error": "Timeout expired",
        }
    except Exception as e:
        return {
            "handshape": handshape,
            "success": False,
            "elapsed_time": time.time() - start_time,
            "error": str(e),
        }


def batch_render(
    handshapes: List[str],
    output_dir: str,
    samples_per_handshape: int,
    parallel_jobs: int = 1,
    model_path: Optional[str] = None,
    image_size: int = 512
) -> Dict:
    """
    Render multiple handshapes in parallel.

    Args:
        handshapes: List of handshape names to render
        output_dir: Output directory
        samples_per_handshape: Samples per handshape
        parallel_jobs: Number of parallel Blender processes
        model_path: Path to MANO model
        image_size: Output image size

    Returns:
        Summary dict with results
    """
    blender_path = find_blender()
    if not blender_path:
        raise RuntimeError(
            "Blender not found. Set BLENDER_PATH environment variable or install Blender."
        )

    print(f"Using Blender: {blender_path}")
    print(f"Rendering {len(handshapes)} handshapes with {samples_per_handshape} samples each")
    print(f"Parallel jobs: {parallel_jobs}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    if parallel_jobs > 1:
        # Parallel rendering
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(
                    render_handshape,
                    blender_path,
                    hs,
                    str(output_path / hs),
                    samples_per_handshape,
                    model_path,
                    image_size
                ): hs for hs in handshapes
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "✓" if result["success"] else "✗"
                print(f"[{status}] {result['handshape']} - {result['elapsed_time']:.1f}s")
    else:
        # Sequential rendering
        for hs in handshapes:
            print(f"Rendering {hs}...")
            result = render_handshape(
                blender_path,
                hs,
                str(output_path / hs),
                samples_per_handshape,
                model_path,
                image_size
            )
            results.append(result)

            status = "✓" if result["success"] else "✗"
            print(f"[{status}] {result['handshape']} - {result['elapsed_time']:.1f}s")

    total_time = time.time() - total_start

    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    summary = {
        "total_handshapes": len(handshapes),
        "successful": successful,
        "failed": failed,
        "total_time": total_time,
        "samples_per_handshape": samples_per_handshape,
        "total_samples": successful * samples_per_handshape,
        "results": results,
    }

    print("-" * 60)
    print(f"Completed: {successful}/{len(handshapes)} handshapes")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total time: {total_time:.1f}s")

    # Save summary
    summary_path = output_path / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Merge metadata from individual renders
    merge_metadata(output_path, handshapes)

    return summary


def merge_metadata(output_dir: Path, handshapes: List[str]):
    """Merge metadata from individual handshape renders."""
    all_samples = []

    for hs in handshapes:
        metadata_path = output_dir / hs / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                all_samples.extend(data.get("samples", []))

    # Create combined metadata
    combined = {
        "version": "1.0",
        "total_samples": len(all_samples),
        "samples": all_samples,
    }

    # Save combined metadata
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(combined, f, indent=2)

    # Create label map
    label_map = {hs: idx for idx, hs in enumerate(sorted(handshapes))}
    with open(output_dir / "label_map.json", 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"Merged metadata: {len(all_samples)} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Batch render ASL handshapes using Blender'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./synthetic_data',
        help='Output directory'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=100,
        help='Samples per handshape'
    )
    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='Number of parallel Blender processes'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to MANO/hand model file'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Output image size'
    )
    parser.add_argument(
        '--alphabet-only',
        action='store_true',
        help='Only render alphabet handshapes (A-Z)'
    )
    parser.add_argument(
        '--handshapes',
        type=str,
        nargs='+',
        help='Specific handshapes to render'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test with single handshape'
    )

    args = parser.parse_args()

    # Determine which handshapes to render
    if args.test:
        handshapes = ["A"]
        args.samples = 3
        print("Running test render (handshape A, 3 samples)")
    elif args.handshapes:
        handshapes = [h.upper() for h in args.handshapes]
    elif args.alphabet_only:
        handshapes = get_alphabet_handshapes()
    else:
        handshapes = get_all_handshapes()

    # Run batch render
    try:
        summary = batch_render(
            handshapes=handshapes,
            output_dir=args.output,
            samples_per_handshape=args.samples,
            parallel_jobs=args.parallel,
            model_path=args.model,
            image_size=args.size
        )

        if summary["failed"] > 0:
            print(f"\nWarning: {summary['failed']} handshapes failed to render")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
