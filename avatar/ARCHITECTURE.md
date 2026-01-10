# Personal Signing Avatar - Architecture

## Vision
User signs on webcam → System captures their appearance + movements → Creates personalized avatar that can sign any ASL phrase

## Implementation Timeline

### 🚀 Weekend MVP (Option A: Face Swap)
**Goal:** User's face + pre-recorded professional signer = personalized signing video

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ User Photo  │ ──► │ Face Extract │ ──► │ Face Swap   │ ──► │ Output Video │
│ (selfie)    │     │ (InsightFace)│     │ (roop)      │     │ (user+signs) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                              ▲
                                              │
                    ┌──────────────────────────┘
                    │
              ┌─────┴─────┐
              │Pre-recorded│
              │Sign Videos │
              │(library)   │
              └───────────┘
```

**Components:**
1. `face_capture.py` - Extract face from selfie/webcam
2. `video_library/` - Pre-recorded sign videos (start with 20 common signs)
3. `face_swap.py` - InsightFace/roop integration
4. `avatar_api.py` - FastAPI endpoint for generation
5. `avatar_ui.html` - Simple web interface

**Time Estimate:** 8-12 hours

---

### 📅 Week 2-3 (Option B: Landmark-Driven)
**Goal:** Extract user's signing motion → Drive avatar with their movements

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ User Video  │ ──► │ MediaPipe    │ ──► │ Landmark    │ ──► │ Retargeted   │
│ (signing)   │     │ Holistic     │     │ Smoothing   │     │ Motion Data  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ Output Video│ ◄── │ Render       │ ◄── │ Avatar Rig  │ ◄── │ Motion       │
│ (avatar)    │     │ (Blender/UE) │     │ (RPM/Custom)│     │ Retarget     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

**Components:**
1. `motion_capture.py` - MediaPipe Holistic extraction
2. `landmark_processor.py` - Smoothing, normalization
3. `avatar_generator.py` - Ready Player Me integration
4. `motion_retarget.py` - Landmarks → avatar bones
5. `avatar_renderer.py` - Blender/Three.js rendering

---

### 🎯 Month 2+ (Production: Generative)
**Goal:** Generate NEW signs the user never recorded

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Text Input  │ ──► │ Sign Gloss   │ ──► │ Motion VAE  │
│ "Hello"     │     │ Lookup       │     │ Generator   │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Video Output│ ◄── │ Neural       │ ◄── │ Style       │
│             │     │ Renderer     │     │ Transfer    │
└─────────────┘     └──────────────┘     └─────────────┘
                                               ▲
                                               │
                                    ┌──────────┴──────────┐
                                    │ User's Appearance   │
                                    │ (face, body, style) │
                                    └─────────────────────┘
```

---

## Weekend MVP - Detailed Plan

### Day 1 (Saturday): Core Pipeline

**Morning (4 hrs):**
- [ ] Set up video library structure
- [ ] Record/collect 20 common sign videos
- [ ] Implement InsightFace face extraction

**Afternoon (4 hrs):**
- [ ] Implement roop face swap
- [ ] Test with sample photos
- [ ] Basic quality validation

### Day 2 (Sunday): API + UI

**Morning (4 hrs):**
- [ ] FastAPI backend for avatar generation
- [ ] Phrase selection endpoint
- [ ] Video generation endpoint

**Afternoon (4 hrs):**
- [ ] Simple web UI (React or vanilla HTML)
- [ ] Upload/webcam photo capture
- [ ] Video download/preview
- [ ] Integration testing

---

## Tech Stack

### Weekend MVP
```
Python 3.10+
├── insightface           # Face analysis/extraction
├── onnxruntime-gpu       # Model inference
├── opencv-python         # Video processing
├── fastapi               # API backend
├── uvicorn               # ASGI server
└── ffmpeg                # Video encoding
```

### Production
```
Python 3.10+
├── mediapipe             # Holistic pose estimation
├── pytorch               # Neural networks
├── trimesh               # 3D mesh processing
├── bpy (Blender)         # 3D rendering
└── three.js              # Web 3D rendering (optional)
```

---

## Pre-recorded Video Library

### Priority 1: Common Greetings (Weekend)
1. HELLO
2. GOODBYE
3. THANK_YOU
4. PLEASE
5. SORRY
6. YES
7. NO
8. HELP
9. I_LOVE_YOU (ILY)
10. NICE_TO_MEET_YOU

### Priority 2: Questions (Week 1)
11. WHAT
12. WHERE
13. WHO
14. WHY
15. HOW
16. WHEN

### Priority 3: Common Words (Week 2)
17. NAME
18. MY
19. YOUR
20. UNDERSTAND
21. AGAIN
22. MORE
23. FINISH
24. WANT
25. NEED

---

## API Design

### Endpoints

```
POST /api/avatar/create
  Body: { photo: base64, name: string }
  Response: { avatar_id: string, preview_url: string }

POST /api/avatar/{id}/sign
  Body: { phrase: string }  // e.g., "HELLO"
  Response: { video_url: string, processing_time: float }

GET /api/phrases
  Response: { phrases: ["HELLO", "THANK_YOU", ...] }

GET /api/avatar/{id}/videos
  Response: { videos: [{ phrase: string, url: string }, ...] }
```

---

## File Structure

```
avatar/
├── ARCHITECTURE.md          # This file
├── requirements.txt         # Dependencies
├── face_capture.py          # Face extraction module
├── face_swap.py             # Face swap using roop/InsightFace
├── video_library/           # Pre-recorded sign videos
│   ├── HELLO.mp4
│   ├── THANK_YOU.mp4
│   └── ...
├── avatar_api.py            # FastAPI backend
├── static/
│   └── avatar_ui.html       # Web interface
└── output/                  # Generated videos
```

---

## Risk Assessment

### Weekend MVP Risks
| Risk | Mitigation |
|------|------------|
| Face swap quality varies | Test multiple face angles, add GFPGAN enhancement |
| Video library too small | Start with 10 most common, expand weekly |
| Processing time too long | Pre-process face embedding, cache results |

### Production Risks
| Risk | Mitigation |
|------|------------|
| MediaPipe hand tracking accuracy | Ensemble with other trackers |
| Avatar uncanny valley | Start with stylized avatars (Ready Player Me) |
| Motion retargeting artifacts | Add temporal smoothing, IK constraints |

---

## Success Metrics

### Weekend Demo
- [ ] Generate personalized signing video in <30 seconds
- [ ] Support 10+ common phrases
- [ ] Works with 80%+ of face photos
- [ ] Shareable video output (MP4)

### Production
- [ ] Real-time avatar animation (<100ms latency)
- [ ] 90%+ phrase coverage
- [ ] Custom avatar appearance matching
- [ ] Two-way communication (sign↔text)
