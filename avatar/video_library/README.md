# Pre-recorded ASL Sign Videos

Place individual sign videos here to enable ASL avatar responses **without** Blender/SMPL-X.

## File naming

- `HELLO.mp4` – sign for "HELLO"
- `HOW.mp4` – sign for "HOW"
- `YOU.mp4` – sign for "YOU"
- etc.

Use uppercase sign names (e.g. `THANK_YOU.mp4`, `NICE_TO_MEET_YOU.mp4`).

## Supported signs (demo fallback)

HELLO, GOODBYE, NICE_TO_MEET_YOU, THANK_YOU, PLEASE, SORRY, YES, NO, WHAT, WHERE, WHO, WHY, HOW, WHEN, I_LOVE_YOU, HAPPY, SAD, UNDERSTAND, HELP, WANT, NEED, LIKE, KNOW, LEARN, FINISH, ME, YOU, MY, YOUR, WE, GOOD, BAD, MORE, AGAIN, EAT, DRINK, SLEEP, WORK, WAIT, STOP, GO, COME, NAME

## Quick test video (optional)

Create a minimal placeholder with ffmpeg:

```bash
# 2-second placeholder (black with "HELLO" text)
ffmpeg -f lavfi -i color=c=black:s=640x480:d=2 -vf "drawtext=text=HELLO:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2:fontcolor=white" -t 2 avatar/video_library/HELLO.mp4
```

Repeat for HOW.mp4, YOU.mp4, etc. to test the sequence.
