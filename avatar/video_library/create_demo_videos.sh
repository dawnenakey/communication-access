#!/bin/bash
# Create minimal placeholder ASL videos for demo (requires ffmpeg)
# Run from project root: bash avatar/video_library/create_demo_videos.sh

set -e
LIB="avatar/video_library"
mkdir -p "$LIB"

# Signs used by demo fallback responses + common avatar response words
SIGNS="HELLO HOW YOU GOOD THANK_YOU YES NICE_TO_MEET_YOU UNDERSTAND HELP NEED WHAT THAT GREAT QUESTION HERE MY ANSWER I AM CAN LEARN COMMUNICATE WELCOME HAPPY TODAY PLEASURE ASSIST THIS APP RECOGNIZES FULL SENTENCES USING AI"

for sign in $SIGNS; do
  out="$LIB/${sign}.mp4"
  if [ -f "$out" ]; then
    echo "Skip $sign (exists)"
  else
    echo "Creating $sign..."
    ffmpeg -y -f lavfi -i "color=c=#1e293b:s=640x480:d=1.5" \
      -vf "drawtext=text=$sign:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2:fontcolor=white" \
      -t 1.5 "$out" 2>/dev/null || true
  fi
done
echo "Done. Add real ASL videos to $LIB/ for production."
