# SignSync AI - ASL Recognition Software

## Original Problem Statement
Build an American Sign Language (ASL) recognition software that reads ASL and translates it into text, audio, and ASL back.

## User Requirements
1. **Input Methods**: Webcam/camera real-time recognition AND video file upload
2. **Recognition**: MediaPipe for hand tracking (free, open-source)
3. **Text-to-Speech**: Browser's built-in speech synthesis
4. **Custom Dictionary**: Upload sign images and map to English words
5. **Authentication**: Google social login (Emergent OAuth)
6. **No Emergent LLM key** required

## Architecture

### Tech Stack
- **Frontend**: React 19 + TailwindCSS + Shadcn UI
- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **Auth**: Emergent Google OAuth
- **Hand Tracking**: MediaPipe Hands (TensorFlow.js)

### Database Collections
- `users` - User profiles
- `user_sessions` - Session management
- `signs` - Custom sign dictionary (word, image, description)
- `translation_history` - Translation records

## User Personas
1. **Deaf/Hard-of-hearing individuals** - Primary users translating ASL to text/speech
2. **ASL Learners** - Using text-to-ASL feature to learn signs
3. **Educators** - Managing custom sign dictionaries for teaching
4. **Accessibility Advocates** - Breaking communication barriers

## Core Requirements (Static)
- [ ] Real-time ASL recognition via webcam
- [ ] Video file upload for batch translation
- [ ] Custom sign dictionary (CRUD operations)
- [ ] Text-to-ASL translation showing sign images
- [ ] Text-to-Speech output (browser synthesis)
- [ ] Translation history storage
- [ ] Google OAuth authentication

## What's Been Implemented ✅
**Date: December 29, 2025**

### Backend (100% functional)
- [x] FastAPI server with /api prefix
- [x] MongoDB integration with proper ObjectId handling
- [x] Google OAuth via Emergent Auth
- [x] Signs CRUD endpoints (POST, GET, PUT, DELETE, SEARCH)
- [x] Translation history endpoints
- [x] Session-based authentication with cookies
- [x] Image upload/storage as base64

### Frontend (95% functional)
- [x] Landing page with hero, features, stats, CTA
- [x] Dashboard with 3 tabs (Camera, Video, Text-to-ASL)
- [x] Webcam integration with start/stop controls
- [x] Video upload functionality
- [x] Text-to-ASL translator
- [x] Sign dictionary management (add/edit/delete)
- [x] Translation history view
- [x] Text-to-Speech integration
- [x] Dark theme with "Cyber-Humanism" aesthetic
- [x] Responsive design

### Mocked Features
- ASL recognition returns random dictionary signs (MediaPipe integration prepared but needs training data)

## Prioritized Backlog

### P0 (Critical) - DONE ✅
- Google OAuth login ✅
- Sign dictionary CRUD ✅
- Basic translation workflow ✅

### P1 (High Priority) - Next Phase
- MediaPipe hand landmark detection integration
- Train classifier on common ASL signs
- Add pre-built ASL sign dataset

### P2 (Medium Priority)
- Video frame-by-frame analysis
- Confidence threshold settings
- Export translation history
- Batch image upload for dictionary

### P3 (Nice to Have)
- Animated avatar for ASL output
- Sign comparison/similarity scoring
- Community dictionary sharing
- Offline mode support

## Next Tasks List
1. Integrate actual MediaPipe hand landmark detection
2. Add pre-built ASL sign images to dictionary
3. Implement sign classifier training pipeline
4. Add video frame extraction for batch processing
5. Enable camera device selection
6. Add accessibility features (screen reader support)

## API Endpoints
- `GET /api/` - API info
- `POST /api/auth/session` - Exchange session_id for token
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout
- `GET /api/signs` - List all signs
- `POST /api/signs` - Create sign (with image upload)
- `GET /api/signs/{sign_id}` - Get sign by ID
- `PUT /api/signs/{sign_id}` - Update sign
- `DELETE /api/signs/{sign_id}` - Delete sign
- `GET /api/signs/search/{word}` - Search signs
- `GET /api/history` - Get user's history
- `POST /api/history` - Create history entry
- `DELETE /api/history/{history_id}` - Delete entry
- `DELETE /api/history` - Clear all history
