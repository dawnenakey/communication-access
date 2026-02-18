#!/usr/bin/env python3
"""
Map your ASL video filenames to gloss format.

Gloss format: UPPERCASE with underscores (e.g., THANK_YOU.mp4, NICE_TO_MEET_YOU.mp4)

Usage:
    python avatar/video_library/map_to_gloss.py --input /path/to/your/videos --output avatar/video_library
    python avatar/video_library/map_to_gloss.py --dry-run  # Show mapping only

Common mappings (add your own to GLOSS_MAP):
    hello.mp4 -> HELLO.mp4
    thank you.mp4 -> THANK_YOU.mp4
    thank_you.mp4 -> THANK_YOU.mp4
"""

import argparse
import re
import shutil
from pathlib import Path

# Map common filename patterns to gloss (add your video names here)
GLOSS_MAP = {
    "hello": "HELLO",
    "hi": "HELLO",
    "how": "HOW",
    "you": "YOU",
    "good": "GOOD",
    "thank you": "THANK_YOU",
    "thankyou": "THANK_YOU",
    "thanks": "THANK_YOU",
    "please": "PLEASE",
    "yes": "YES",
    "no": "NO",
    "help": "HELP",
    "sorry": "SORRY",
    "love": "LOVE",
    "friend": "FRIEND",
    "family": "FAMILY",
    "eat": "EAT",
    "drink": "DRINK",
    "water": "WATER",
    "home": "HOME",
    "work": "WORK",
    "school": "SCHOOL",
    "name": "NAME",
    "bad": "BAD",
    "want": "WANT",
    "need": "NEED",
    "like": "LIKE",
    "understand": "UNDERSTAND",
    "learn": "LEARN",
    "know": "KNOW",
    "think": "THINK",
    "see": "SEE",
    "hear": "HEAR",
    "nice to meet you": "NICE_TO_MEET_YOU",
    "what": "WHAT",
    "where": "WHERE",
    "who": "WHO",
    "why": "WHY",
    "when": "WHEN",
    "i": "I",
    "am": "AM",
    "can": "CAN",
    "this": "THIS",
    "that": "THAT",
    "the": "THE",
    "a": "A",
    "in": "IN",
    "sign": "SIGN",
    "language": "LANGUAGE",
    "app": "APP",
    "recognizes": "RECOGNIZES",
    "full": "FULL",
    "sentences": "SENTENCES",
    "using": "USING",
    "ai": "AI",
}


def filename_to_gloss(name: str) -> str:
    """Convert filename (without extension) to gloss format."""
    base = Path(name).stem.lower()
    # Try exact match first
    if base in GLOSS_MAP:
        return GLOSS_MAP[base]
    # Try with spaces normalized to underscores
    normalized = re.sub(r"[\s\-]+", "_", base).strip("_")
    if normalized in GLOSS_MAP:
        return GLOSS_MAP[normalized]
    # Try with underscores as spaces
    as_spaces = base.replace("_", " ")
    if as_spaces in GLOSS_MAP:
        return GLOSS_MAP[as_spaces]
    # Default: uppercase with underscores
    return base.upper().replace(" ", "_").replace("-", "_")


def main():
    ap = argparse.ArgumentParser(description="Map ASL videos to gloss filenames")
    ap.add_argument("--input", "-i", type=str, default=".", help="Input directory with your videos")
    ap.add_argument("--output", "-o", type=str, default="avatar/video_library", help="Output directory")
    ap.add_argument("--dry-run", action="store_true", help="Show mapping only, don't copy")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))
    if not videos:
        print(f"No .mp4 files found in {input_dir}")
        return 1

    print("Mapping your videos to gloss format:\n")
    for v in sorted(videos):
        gloss = filename_to_gloss(v.name)
        dest = output_dir / f"{gloss}.mp4"
        print(f"  {v.name} -> {gloss}.mp4")
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v, dest)
            print(f"    Copied to {dest}")

    if args.dry_run:
        print("\nRun without --dry-run to copy files.")
    else:
        print(f"\nDone. {len(videos)} videos copied to {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
