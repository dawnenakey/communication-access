# SonZo AI - Investor Demo Guide

## Quick Start (5 minutes to demo-ready)

### 1. Start Demo Mode
```bash
cd /path/to/communication-access
python launch.py --demo
```

This starts all services with **simulated recognition** - no ML model needed!

### 2. Open the App
Navigate to: **http://localhost:8081**

### 3. Demo Flow

---

## Demo Script (3-5 minutes)

### Opening (30 seconds)
> "SonZo AI breaks down communication barriers between Deaf and hearing communities.
> Let me show you how it works."

### Part 1: Onboarding (45 seconds)
1. Click **"Get Started"**
2. Upload or take a photo for avatar
3. Skip calibration (or demo it)
4. Select "Learn ASL" as goal
5. Complete preferences

> "New users create their personal signing avatar in under 30 seconds."

### Part 2: Real-Time Recognition (60 seconds)
1. Click **"Quick Sign"** button
2. Hold up hand in front of camera
3. Show recognition ring turning green
4. Demonstrate a few signs:
   - Wave (HELLO)
   - Thumbs up (GOOD)
   - Open hand (STOP/WAIT)

> "Our computer vision recognizes signs in real-time with under 500ms latency.
> The visual feedback ring shows the user exactly what's happening."

### Part 3: Avatar Response (45 seconds)
1. Type "How are you?" in the input
2. Show avatar signing back
3. Demonstrate text → sign translation

> "Users can type any phrase and their personal avatar signs it back.
> This enables two-way communication - Deaf users can sign, hearing users can type."

### Part 4: Progress & Gamification (30 seconds)
1. Open Dashboard
2. Show signs learned, streak, achievements
3. Show daily challenge

> "We've built Duolingo-style gamification to keep users engaged.
> Daily streaks, achievements, and XP make learning fun."

### Part 5: Accessibility (30 seconds)
1. Go to Settings
2. Toggle Dark Mode
3. Toggle High Contrast Mode
4. Show left-handed mode option

> "Accessibility isn't an afterthought - it's built into every feature.
> High contrast mode, large text, screen reader support, and more."

### Closing (30 seconds)
> "SonZo AI is the first app that provides:
> 1. Real-time sign recognition
> 2. Personal signing avatars
> 3. Two-way communication
> 4. Gamified learning
>
> All in one mobile-first, accessibility-first platform."

---

## Technical Demo Points

### For Technical Investors
- "Recognition runs at 30 FPS with <500ms latency"
- "3D CNN + LSTM + Attention architecture for sign recognition"
- "Face swap uses InsightFace + GFPGAN for avatar generation"
- "Synthetic training data generated via Blender MANO pipeline"

### For Business Investors
- "70 million Deaf individuals worldwide"
- "$1.2B market for sign language education"
- "No major competitor offers real-time recognition + avatars"
- "B2B potential: healthcare, education, government services"

---

## Recording Your Demo

### Setup
1. Use OBS Studio or QuickTime for screen recording
2. Set resolution to 1920x1080
3. Include webcam overlay (small, bottom-right)
4. Use external mic for clear audio

### Recording Checklist
- [ ] Close unnecessary tabs/apps
- [ ] Put phone on silent
- [ ] Test webcam is working
- [ ] Test audio levels
- [ ] Run through demo once before recording

### Export Settings
- Format: MP4
- Codec: H.264
- Resolution: 1080p
- Frame rate: 30 FPS
- Audio: AAC 128kbps

---

## Troubleshooting

### "Camera not working"
- Check browser permissions (chrome://settings/content/camera)
- Try a different browser

### "Recognition not responding"
- Refresh the page
- Check terminal for errors
- Restart with `python launch.py --demo`

### "Avatar not loading"
- Ensure photo was uploaded successfully
- Check network tab for API errors
- Avatar API runs on port 8080

---

## Demo Environment Checklist

Before the demo:
- [ ] Laptop charged
- [ ] Good lighting for webcam
- [ ] Stable internet connection
- [ ] Demo mode running (`python launch.py --demo`)
- [ ] Browser cache cleared
- [ ] No notifications enabled
- [ ] Backup demo video ready (in case of issues)

---

## Contact

**Dawnena Key**
Founder & CEO, SonZo AI
dawnena@sonzo.ai

Patent Pending | Made with ❤️ for the Deaf community
