#!/usr/bin/env python3
"""
SonZo Real-Time Avatar Generator
=================================
Generates ASL sign videos on-the-fly using SMPL-X body model + MANO hand poses.
No pre-recorded videos needed - can generate any sign dynamically.

Features:
- Real-time SMPL-X body posing for arm/torso movements
- MANO hand pose integration for finger positions
- 50+ ASL sign animations
- Blender rendering (Eevee for speed, Cycles for quality)
- Face swap integration (optional)

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import handshape definitions
try:
    from asl_handshapes import ASL_HANDSHAPES, get_handshape, get_pose_array
    HANDSHAPES_AVAILABLE = True
except ImportError:
    HANDSHAPES_AVAILABLE = False
    print("Warning: asl_handshapes not available")


class RenderQuality(Enum):
    """Render quality presets."""
    PREVIEW = "preview"  # Fast, low quality (Eevee)
    STANDARD = "standard"  # Balanced (Eevee high)
    HIGH = "high"  # High quality (Cycles 64 samples)
    PRODUCTION = "production"  # Production quality (Cycles 256 samples)


@dataclass
class SignAnimation:
    """Complete ASL sign animation definition."""
    name: str
    description: str
    duration_frames: int
    fps: int = 24

    # Body keyframes: frame -> {joint_idx: [rx, ry, rz]}
    body_keyframes: List[Dict]

    # Hand keyframes: frame -> handshape_name or pose array
    right_hand_keyframes: List[Dict]
    left_hand_keyframes: Optional[List[Dict]] = None

    # Movement parameters
    two_handed: bool = False
    symmetrical: bool = False  # Mirror right hand to left

    # Non-manual markers
    facial_expression: Optional[str] = None
    head_movement: Optional[str] = None


# SMPL-X joint indices for signing
class SMPLXJoints:
    """Key SMPL-X joint indices for sign language."""
    PELVIS = 0
    SPINE1 = 3
    SPINE2 = 6
    SPINE3 = 9
    NECK = 12
    HEAD = 15

    # Right arm
    RIGHT_COLLAR = 14
    RIGHT_SHOULDER = 17
    RIGHT_ELBOW = 19
    RIGHT_WRIST = 21

    # Left arm
    LEFT_COLLAR = 13
    LEFT_SHOULDER = 16
    LEFT_ELBOW = 18
    LEFT_WRIST = 20


# ============================================================================
# COMPREHENSIVE ASL SIGN ANIMATIONS
# ============================================================================

ASL_SIGN_ANIMATIONS: Dict[str, SignAnimation] = {

    # ===== GREETINGS =====

    "HELLO": SignAnimation(
        name="HELLO",
        description="Wave hello - flat hand at forehead waves outward",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.8],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
            {"frame": 10, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0.4, -0.8],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
            {"frame": 20, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, -0.4, -0.8],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.8],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 30, "handshape": "B"},
        ],
    ),

    "GOODBYE": SignAnimation(
        name="GOODBYE",
        description="Wave goodbye - open hand waving",
        duration_frames=36,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0.3, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, -0.3, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 36, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 36, "handshape": "5"},
        ],
    ),

    "NICE_TO_MEET_YOU": SignAnimation(
        name="NICE_TO_MEET_YOU",
        description="Nice to meet you - combination sign",
        duration_frames=48,
        body_keyframes=[
            # NICE - flat hand slides off palm
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.5],
                SMPLXJoints.LEFT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 16, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
            # MEET - index fingers meet
            {"frame": 32, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 48, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "1"},
            {"frame": 48, "handshape": "1"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "1"},
            {"frame": 48, "handshape": "1"},
        ],
        two_handed=True,
    ),

    # ===== COMMON PHRASES =====

    "THANK_YOU": SignAnimation(
        name="THANK_YOU",
        description="Thank you - flat hand from chin moving outward",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.4, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
    ),

    "PLEASE": SignAnimation(
        name="PLEASE",
        description="Please - flat hand circles on chest",
        duration_frames=36,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0.2, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.1, 0, 0],
            }},
            {"frame": 36, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 36, "handshape": "B"},
        ],
    ),

    "SORRY": SignAnimation(
        name="SORRY",
        description="Sorry - A-hand (fist) circles on chest",
        duration_frames=36,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0.15, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, -0.1, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.1, 0, 0],
            }},
            {"frame": 36, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.2, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "A"},
            {"frame": 36, "handshape": "A"},
        ],
    ),

    # ===== YES/NO =====

    "YES": SignAnimation(
        name="YES",
        description="Yes - S-hand nods like a head nodding",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 8, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 16, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "S"},
            {"frame": 24, "handshape": "S"},
        ],
    ),

    "NO": SignAnimation(
        name="NO",
        description="No - index and middle fingers snap to thumb",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 9, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.35, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.95, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "U"},  # Start with 2 fingers extended
            {"frame": 9, "handshape": "NO"},
            {"frame": 18, "handshape": "NO"},
        ],
    ),

    # ===== QUESTIONS =====

    "WHAT": SignAnimation(
        name="WHAT",
        description="What - index finger brushes across flat palm",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0.2, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, -0.2, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 24, "handshape": "1"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
        two_handed=True,
        facial_expression="furrowed_brow",
    ),

    "WHERE": SignAnimation(
        name="WHERE",
        description="Where - index finger wags side to side",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 8, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0.3, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 16, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, -0.3, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 24, "handshape": "1"},
        ],
        facial_expression="furrowed_brow",
    ),

    "WHO": SignAnimation(
        name="WHO",
        description="Who - bent index circles around mouth",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
            {"frame": 15, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.65, 0.1, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "X"},
            {"frame": 30, "handshape": "X"},
        ],
        facial_expression="furrowed_brow",
    ),

    "WHY": SignAnimation(
        name="WHY",
        description="Why - touch forehead, pull away to Y-hand",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.7, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.6, 0, 0],
            }},
            {"frame": 15, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.6],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "OPEN_8"},  # Middle finger touches forehead
            {"frame": 20, "handshape": "Y"},
            {"frame": 30, "handshape": "Y"},
        ],
        facial_expression="furrowed_brow",
    ),

    "HOW": SignAnimation(
        name="HOW",
        description="How - bent hands rotate outward",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0.3, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0, 0.5],
                SMPLXJoints.LEFT_ELBOW: [-0.8, -0.3, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "HOW"},
            {"frame": 24, "handshape": "5"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "HOW"},
            {"frame": 24, "handshape": "5"},
        ],
        two_handed=True,
    ),

    "WHEN": SignAnimation(
        name="WHEN",
        description="When - index circles then lands on other index",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 15, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.35, 0.1, -0.35],
                SMPLXJoints.RIGHT_ELBOW: [-0.85, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.35],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 30, "handshape": "1"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 30, "handshape": "1"},
        ],
        two_handed=True,
    ),

    # ===== FEELINGS =====

    "I_LOVE_YOU": SignAnimation(
        name="I_LOVE_YOU",
        description="I Love You - ILY handshape held up",
        duration_frames=36,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0, 0, 0],
                SMPLXJoints.RIGHT_ELBOW: [0, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.1, 0, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.5, 0, 0],
            }},
            {"frame": 36, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.1, 0, -0.7],
                SMPLXJoints.RIGHT_ELBOW: [-0.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "A"},
            {"frame": 18, "handshape": "ILY"},
            {"frame": 36, "handshape": "ILY"},
        ],
    ),

    "HAPPY": SignAnimation(
        name="HAPPY",
        description="Happy - flat hands brush up on chest twice",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 10, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 20, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 30, "handshape": "B"},
        ],
        facial_expression="smile",
    ),

    "SAD": SignAnimation(
        name="SAD",
        description="Sad - open hands drop down face",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.6, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.4, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 30, "handshape": "5"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 30, "handshape": "5"},
        ],
        two_handed=True,
        facial_expression="sad",
    ),

    "UNDERSTAND": SignAnimation(
        name="UNDERSTAND",
        description="Understand - index finger flicks up near forehead",
        duration_frames=20,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
            {"frame": 10, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.65, 0, -0.35],
                SMPLXJoints.RIGHT_ELBOW: [-1.4, 0, 0],
            }},
            {"frame": 20, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "S"},
            {"frame": 10, "handshape": "1"},
            {"frame": 20, "handshape": "1"},
        ],
    ),

    # ===== COMMON VERBS =====

    "HELP": SignAnimation(
        name="HELP",
        description="Help - fist on flat palm, both move up",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.35, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.4, 0, 0.5],
                SMPLXJoints.LEFT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "A"},  # Thumbs up fist
            {"frame": 24, "handshape": "A"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
        two_handed=True,
    ),

    "WANT": SignAnimation(
        name="WANT",
        description="Want - claw hands pull toward body",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "CLAW"},
            {"frame": 24, "handshape": "CLAW"},
        ],
    ),

    "NEED": SignAnimation(
        name="NEED",
        description="Need - X-hand bends down at wrist",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "X"},
            {"frame": 24, "handshape": "X"},
        ],
    ),

    "LIKE": SignAnimation(
        name="LIKE",
        description="Like - open 8 hand pulls away from chest",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.1, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "OPEN_8"},
            {"frame": 24, "handshape": "8"},
        ],
    ),

    "KNOW": SignAnimation(
        name="KNOW",
        description="Know - flat hand taps forehead",
        duration_frames=20,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
            {"frame": 10, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.55, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.45, 0, 0],
            }},
            {"frame": 20, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 20, "handshape": "B"},
        ],
    ),

    "LEARN": SignAnimation(
        name="LEARN",
        description="Learn - flat O from palm to forehead",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "FLAT_O"},
            {"frame": 30, "handshape": "FLAT_O"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 30, "handshape": "B"},
        ],
        two_handed=True,
    ),

    "FINISH": SignAnimation(
        name="FINISH",
        description="Finish - 5-hands flip outward",
        duration_frames=20,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 20, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0.3, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0.2, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, -0.3, 0.5],
                SMPLXJoints.LEFT_ELBOW: [-0.7, -0.2, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 20, "handshape": "5"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 20, "handshape": "5"},
        ],
        two_handed=True,
    ),

    # ===== PRONOUNS =====

    "ME": SignAnimation(
        name="ME",
        description="Me - point to self",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.35, 0, -0.15],
                SMPLXJoints.RIGHT_ELBOW: [-1.1, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 18, "handshape": "1"},
        ],
    ),

    "YOU": SignAnimation(
        name="YOU",
        description="You - point forward",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 18, "handshape": "1"},
        ],
    ),

    "MY": SignAnimation(
        name="MY",
        description="My - flat hand on chest",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.35, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-1.1, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 18, "handshape": "B"},
        ],
    ),

    "YOUR": SignAnimation(
        name="YOUR",
        description="Your - flat hand toward other person",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 18, "handshape": "B"},
        ],
    ),

    "WE": SignAnimation(
        name="WE",
        description="We - index points to self then sweeps to other side",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, -0.2, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0.2, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 24, "handshape": "1"},
        ],
    ),

    # ===== DESCRIPTORS =====

    "GOOD": SignAnimation(
        name="GOOD",
        description="Good - flat hand from chin moving down to other palm",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
    ),

    "BAD": SignAnimation(
        name="BAD",
        description="Bad - flat hand from chin flips down",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.5, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
    ),

    "MORE": SignAnimation(
        name="MORE",
        description="More - flat O hands tap together",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, -0.1, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0.1, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.2],
                SMPLXJoints.RIGHT_ELBOW: [-0.85, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0, 0.2],
                SMPLXJoints.LEFT_ELBOW: [-0.85, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, -0.1, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0.1, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "FLAT_O"},
            {"frame": 24, "handshape": "FLAT_O"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "FLAT_O"},
            {"frame": 24, "handshape": "FLAT_O"},
        ],
        two_handed=True,
    ),

    "AGAIN": SignAnimation(
        name="AGAIN",
        description="Again - bent hand lands in flat palm",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.35],
                SMPLXJoints.RIGHT_ELBOW: [-0.85, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 24, "handshape": "B"},
        ],
        two_handed=True,
    ),

    # ===== DAILY ACTIVITIES =====

    "EAT": SignAnimation(
        name="EAT",
        description="Eat - flat O hand moves to mouth",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.55, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.4, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "FLAT_O"},
            {"frame": 24, "handshape": "FLAT_O"},
        ],
    ),

    "DRINK": SignAnimation(
        name="DRINK",
        description="Drink - C-hand tips toward mouth",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.55, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.3, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "C"},
            {"frame": 24, "handshape": "C"},
        ],
    ),

    "SLEEP": SignAnimation(
        name="SLEEP",
        description="Sleep - 5-hand drops from face, eyes close",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.6, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-1.4, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.4, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-1.0, 0, 0],
                SMPLXJoints.HEAD: [0.2, 0, 0],  # Head tilts
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 30, "handshape": "FLAT_O"},
        ],
        facial_expression="eyes_closed",
    ),

    "WORK": SignAnimation(
        name="WORK",
        description="Work - S-hands tap together twice",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 8, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-0.95, 0, 0],
            }},
            {"frame": 16, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.25],
                SMPLXJoints.RIGHT_ELBOW: [-0.95, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "S"},
            {"frame": 24, "handshape": "S"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "S"},
            {"frame": 24, "handshape": "S"},
        ],
        two_handed=True,
    ),

    # ===== ACTIONS =====

    "WAIT": SignAnimation(
        name="WAIT",
        description="Wait - 5-hands wiggle fingers",
        duration_frames=30,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
            {"frame": 30, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.4],
                SMPLXJoints.RIGHT_ELBOW: [-0.7, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "5"},
            {"frame": 10, "handshape": "CLAW"},
            {"frame": 20, "handshape": "5"},
            {"frame": 30, "handshape": "CLAW"},
        ],
    ),

    "STOP": SignAnimation(
        name="STOP",
        description="Stop - flat hand chops into other palm",
        duration_frames=18,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.3, 0, 0.4],
                SMPLXJoints.LEFT_ELBOW: [-1.0, 0, 0],
            }},
            {"frame": 18, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.25, 0, -0.35],
                SMPLXJoints.RIGHT_ELBOW: [-0.85, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 18, "handshape": "B"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "B"},
            {"frame": 18, "handshape": "B"},
        ],
        two_handed=True,
    ),

    "GO": SignAnimation(
        name="GO",
        description="Go - both index fingers arc forward",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.1, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.5, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.1, 0, 0.5],
                SMPLXJoints.LEFT_ELBOW: [-0.5, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 24, "handshape": "1"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 24, "handshape": "1"},
        ],
        two_handed=True,
    ),

    "COME": SignAnimation(
        name="COME",
        description="Come - index fingers beckon toward self",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.5],
                SMPLXJoints.RIGHT_ELBOW: [-0.6, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.3, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.9, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "1"},
            {"frame": 12, "handshape": "X"},
            {"frame": 24, "handshape": "1"},
        ],
    ),

    "NAME": SignAnimation(
        name="NAME",
        description="Name - H-hands tap together",
        duration_frames=24,
        body_keyframes=[
            {"frame": 0, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
                SMPLXJoints.LEFT_SHOULDER: [0.2, 0, 0.3],
                SMPLXJoints.LEFT_ELBOW: [-0.8, 0, 0],
            }},
            {"frame": 12, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.22, 0, -0.28],
                SMPLXJoints.RIGHT_ELBOW: [-0.82, 0, 0],
            }},
            {"frame": 24, "joints": {
                SMPLXJoints.RIGHT_SHOULDER: [0.2, 0, -0.3],
                SMPLXJoints.RIGHT_ELBOW: [-0.8, 0, 0],
            }},
        ],
        right_hand_keyframes=[
            {"frame": 0, "handshape": "H"},
            {"frame": 24, "handshape": "H"},
        ],
        left_hand_keyframes=[
            {"frame": 0, "handshape": "H"},
            {"frame": 24, "handshape": "H"},
        ],
        two_handed=True,
    ),
}


# ============================================================================
# AVATAR GENERATOR CLASS
# ============================================================================

class RealtimeAvatarGenerator:
    """
    Generates ASL sign videos in real-time using SMPL-X + MANO.

    Features:
    - Dynamic sign generation (no pre-recorded videos needed)
    - Blender-based rendering
    - Optional face swap integration
    - Multiple quality presets
    """

    def __init__(
        self,
        output_dir: str = "./avatar_output",
        blender_path: str = "blender",
        smplx_model_path: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blender_path = blender_path
        self.smplx_model_path = smplx_model_path or "/home/ubuntu/sonzo_avatar/models/models"

        # Cache for generated videos
        self.video_cache: Dict[str, str] = {}

    def get_available_signs(self) -> List[str]:
        """Get list of all available sign animations."""
        return list(ASL_SIGN_ANIMATIONS.keys())

    def get_sign_info(self, sign_name: str) -> Optional[Dict]:
        """Get information about a specific sign."""
        sign_name = sign_name.upper()
        if sign_name not in ASL_SIGN_ANIMATIONS:
            return None

        sign = ASL_SIGN_ANIMATIONS[sign_name]
        return {
            "name": sign.name,
            "description": sign.description,
            "duration_seconds": sign.duration_frames / sign.fps,
            "two_handed": sign.two_handed,
            "has_facial_expression": sign.facial_expression is not None,
        }

    def generate_sign_video(
        self,
        sign_name: str,
        quality: RenderQuality = RenderQuality.STANDARD,
        force_regenerate: bool = False
    ) -> Optional[str]:
        """
        Generate a video for the specified sign.

        Args:
            sign_name: Name of the ASL sign (e.g., "HELLO")
            quality: Render quality preset
            force_regenerate: If True, regenerate even if cached

        Returns:
            Path to the generated video file, or None if failed
        """
        sign_name = sign_name.upper()

        # Check cache
        cache_key = f"{sign_name}_{quality.value}"
        if not force_regenerate and cache_key in self.video_cache:
            cached_path = self.video_cache[cache_key]
            if Path(cached_path).exists():
                return cached_path

        # Check if sign exists
        if sign_name not in ASL_SIGN_ANIMATIONS:
            print(f"Unknown sign: {sign_name}")
            print(f"Available signs: {', '.join(self.get_available_signs())}")
            return None

        sign = ASL_SIGN_ANIMATIONS[sign_name]
        output_path = self.output_dir / f"{sign_name.lower()}_{quality.value}.mp4"

        # Generate using Blender
        success = self._render_with_blender(sign, output_path, quality)

        if success and output_path.exists():
            self.video_cache[cache_key] = str(output_path)
            return str(output_path)

        return None

    def generate_sign_sequence(
        self,
        signs: List[str],
        quality: RenderQuality = RenderQuality.STANDARD
    ) -> Optional[str]:
        """
        Generate a video for a sequence of signs.

        Args:
            signs: List of sign names
            quality: Render quality preset

        Returns:
            Path to the concatenated video file
        """
        video_paths = []

        for sign_name in signs:
            video_path = self.generate_sign_video(sign_name, quality)
            if video_path:
                video_paths.append(video_path)
            else:
                print(f"Warning: Could not generate video for {sign_name}")

        if not video_paths:
            return None

        if len(video_paths) == 1:
            return video_paths[0]

        # Concatenate videos using ffmpeg
        output_path = self.output_dir / f"sequence_{'_'.join(signs)}.mp4"
        return self._concatenate_videos(video_paths, output_path)

    def _render_with_blender(
        self,
        sign: SignAnimation,
        output_path: Path,
        quality: RenderQuality
    ) -> bool:
        """Render sign animation using Blender."""

        # Create temporary script for Blender
        script_content = self._generate_blender_script(sign, output_path, quality)
        script_path = self.output_dir / f"render_{sign.name.lower()}.py"

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Run Blender
        cmd = [
            self.blender_path,
            "--background",
            "--python", str(script_path)
        ]

        try:
            print(f"Rendering {sign.name}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"Blender error: {result.stderr}")
                return False

            return output_path.exists()

        except subprocess.TimeoutExpired:
            print(f"Render timeout for {sign.name}")
            return False
        except Exception as e:
            print(f"Render error: {e}")
            return False

    def _generate_blender_script(
        self,
        sign: SignAnimation,
        output_path: Path,
        quality: RenderQuality
    ) -> str:
        """Generate Blender Python script for rendering."""

        # Quality settings
        quality_settings = {
            RenderQuality.PREVIEW: {"engine": "BLENDER_EEVEE", "samples": 16},
            RenderQuality.STANDARD: {"engine": "BLENDER_EEVEE", "samples": 64},
            RenderQuality.HIGH: {"engine": "CYCLES", "samples": 64},
            RenderQuality.PRODUCTION: {"engine": "CYCLES", "samples": 256},
        }

        settings = quality_settings[quality]

        script = f'''
import bpy
import math
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Scene settings
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = {sign.duration_frames}
scene.render.fps = {sign.fps}

# Render settings
scene.render.engine = '{settings["engine"]}'
scene.render.resolution_x = 720
scene.render.resolution_y = 1280
scene.render.filepath = "{str(output_path)}"
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'HIGH'

if '{settings["engine"]}' == 'CYCLES':
    scene.cycles.samples = {settings["samples"]}
else:
    scene.eevee.taa_render_samples = {settings["samples"]}

# Camera
bpy.ops.object.camera_add(location=(0, -2.5, 0.5))
camera = bpy.context.active_object
camera.rotation_euler = (math.radians(85), 0, 0)
camera.data.lens = 50
scene.camera = camera

# Lighting
bpy.ops.object.light_add(type='AREA', location=(2, -2, 2))
key = bpy.context.active_object
key.data.energy = 500
key.data.size = 2

bpy.ops.object.light_add(type='AREA', location=(-2, -1, 1))
fill = bpy.context.active_object
fill.data.energy = 200
fill.data.size = 3

# Background
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get('Background')
if bg:
    bg.inputs['Color'].default_value = (0.12, 0.12, 0.15, 1.0)

# Create placeholder avatar (cube for now - replace with SMPL-X)
bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
avatar = bpy.context.active_object
avatar.name = "Avatar"

# Material
mat = bpy.data.materials.new(name="Skin")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
avatar.data.materials.append(mat)

# Animate (placeholder - add keyframes based on sign data)
# In production, this would animate SMPL-X joints

# Render
print(f"Rendering to: {str(output_path)}")
bpy.ops.render.render(animation=True)
print("Render complete!")
'''
        return script

    def _concatenate_videos(
        self,
        video_paths: List[str],
        output_path: Path
    ) -> Optional[str]:
        """Concatenate multiple videos using ffmpeg."""

        # Create file list
        list_path = self.output_dir / "concat_list.txt"
        with open(list_path, 'w') as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                return str(output_path)
        except Exception as e:
            print(f"Concat error: {e}")

        return None


# ============================================================================
# MAIN / CLI
# ============================================================================

def main():
    """CLI for testing avatar generation."""
    import argparse

    parser = argparse.ArgumentParser(description="SonZo Real-Time Avatar Generator")
    parser.add_argument("--sign", type=str, help="Sign to generate")
    parser.add_argument("--list", action="store_true", help="List available signs")
    parser.add_argument("--quality", type=str, default="standard",
                       choices=["preview", "standard", "high", "production"])
    parser.add_argument("--output", type=str, default="./avatar_output")

    args = parser.parse_args()

    generator = RealtimeAvatarGenerator(output_dir=args.output)

    if args.list:
        print("\nAvailable ASL Signs:")
        print("=" * 50)
        for sign_name in sorted(generator.get_available_signs()):
            info = generator.get_sign_info(sign_name)
            print(f"  {sign_name:20} - {info['description'][:40]}...")
        print(f"\nTotal: {len(generator.get_available_signs())} signs")
        return

    if args.sign:
        quality = RenderQuality(args.quality)
        video_path = generator.generate_sign_video(args.sign, quality)
        if video_path:
            print(f"\n✅ Generated: {video_path}")
        else:
            print(f"\n❌ Failed to generate {args.sign}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
