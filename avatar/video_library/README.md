# Pre-recorded ASL Sign Videos

Place individual sign videos here to enable ASL avatar responses **without** Blender/SMPL-X.

## Important: Demo vs Real Signing

The `create_demo_videos.sh` script creates **placeholder videos** only – colored rectangles with text labels. These are for testing the pipeline, not for real ASL output.

**For real signing**, replace the placeholder `.mp4` files with actual videos of someone performing each sign:

- **File naming**: `HELLO.mp4`, `HOW.mp4`, `YOU.mp4`, `THANK_YOU.mp4`, etc. (uppercase, underscores)
- **Format**: MP4, 640×480 or similar, ~1–2 seconds per sign
- **Sources**: Record your own, or use licensed ASL video datasets (e.g. ASLLVD)

## File naming

- `HELLO.mp4` – sign for "HELLO"
- `HOW.mp4` – sign for "HOW"
- `YOU.mp4` – sign for "YOU"
- etc.

Use uppercase sign names (e.g. `THANK_YOU.mp4`, `NICE_TO_MEET_YOU.mp4`).

## Supported signs (demo fallback)

HELLO, GOODBYE, NICE_TO_MEET_YOU, THANK_YOU, PLEASE, SORRY, YES, NO, WHAT, WHERE, WHO, WHY, HOW, WHEN, I_LOVE_YOU, HAPPY, SAD, UNDERSTAND, HELP, WANT, NEED, LIKE, KNOW, LEARN, FINISH, ME, YOU, MY, YOUR, WE, GOOD, BAD, MORE, AGAIN, EAT, DRINK, SLEEP, WORK, WAIT, STOP, GO, COME, NAME

## Quick test video (placeholder only)

```bash
# Creates placeholder (text on colored background) – NOT real signing
bash avatar/video_library/create_demo_videos.sh
```

To get real signing: replace the generated files with real ASL videos.
