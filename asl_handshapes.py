"""
ASL Handshape Definitions as MANO/Blender Pose Parameters
=========================================================
Each handshape is defined by finger joint rotations in radians.
Joint order: Index(3), Middle(3), Ring(3), Pinky(3), Thumb(3) = 15 joints
Each joint has [flexion, abduction, rotation] angles.

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FingerJoint(Enum):
    """MANO joint indices for each finger."""
    # Index finger
    INDEX_MCP = 0
    INDEX_PIP = 1
    INDEX_DIP = 2
    # Middle finger
    MIDDLE_MCP = 3
    MIDDLE_PIP = 4
    MIDDLE_DIP = 5
    # Ring finger
    RING_MCP = 6
    RING_PIP = 7
    RING_DIP = 8
    # Pinky
    PINKY_MCP = 9
    PINKY_PIP = 10
    PINKY_DIP = 11
    # Thumb
    THUMB_CMC = 12
    THUMB_MCP = 13
    THUMB_IP = 14


@dataclass
class HandshapeConfig:
    """Configuration for a single ASL handshape."""
    name: str
    description: str
    pose: List[List[float]]  # 15 joints x 3 angles
    dominant_hand: str = "right"
    two_handed: bool = False
    non_manual_markers: Optional[List[str]] = None  # Facial expressions
    movement_required: bool = False


# Common pose patterns for reuse
FULLY_FLEXED = [1.5, 0.0, 0.0]  # Finger curled into palm
PARTIALLY_FLEXED = [0.8, 0.0, 0.0]
SLIGHTLY_BENT = [0.3, 0.0, 0.0]
EXTENDED = [0.0, 0.0, 0.0]  # Finger straight out
SPREAD_EXTENDED = [0.0, 0.3, 0.0]  # Extended with spread

# Thumb positions
THUMB_ALONGSIDE = [[0.3, 0.5, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]
THUMB_ACROSS_PALM = [[0.8, 1.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]]
THUMB_EXTENDED = [[0.0, -0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
THUMB_TUCKED = [[1.0, 0.8, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]]
THUMB_TOUCHING_INDEX = [[0.5, 0.6, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]]
THUMB_UP = [[-0.2, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def make_pose(
    index: List[List[float]],
    middle: List[List[float]],
    ring: List[List[float]],
    pinky: List[List[float]],
    thumb: List[List[float]]
) -> List[List[float]]:
    """Helper to construct a full pose from finger configurations."""
    return index + middle + ring + pinky + thumb


# ============================================================================
# ASL ALPHABET HANDSHAPES
# ============================================================================

ASL_HANDSHAPES: Dict[str, HandshapeConfig] = {
    
    # ===== A =====
    "A": HandshapeConfig(
        name="A",
        description="Fist with thumb alongside index finger",
        pose=make_pose(
            index=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_ALONGSIDE
        )
    ),
    
    # ===== B =====
    "B": HandshapeConfig(
        name="B",
        description="Flat hand, fingers together extended, thumb across palm",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=THUMB_ACROSS_PALM
        )
    ),
    
    # ===== C =====
    "C": HandshapeConfig(
        name="C",
        description="Curved hand forming C shape",
        pose=make_pose(
            index=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            middle=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            ring=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            pinky=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            thumb=[[0.4, 0.3, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]
        )
    ),
    
    # ===== D =====
    "D": HandshapeConfig(
        name="D",
        description="Index pointing up, other fingers curled to touch thumb",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.0, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.0, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.0, 0.0, 0.0]],
            thumb=THUMB_TOUCHING_INDEX
        )
    ),
    
    # ===== E =====
    "E": HandshapeConfig(
        name="E",
        description="Fingers curled, thumb tucked under fingertips",
        pose=make_pose(
            index=[[1.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            middle=[[1.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            ring=[[1.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            pinky=[[1.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== F =====
    "F": HandshapeConfig(
        name="F",
        description="Thumb and index touch to form circle, other fingers extended",
        pose=make_pose(
            index=[[0.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0]],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=[[0.6, 0.5, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
    ),
    
    # ===== G =====
    "G": HandshapeConfig(
        name="G",
        description="Index and thumb extended parallel, other fingers curled",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ),
    
    # ===== H =====
    "H": HandshapeConfig(
        name="H",
        description="Index and middle extended together horizontally",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== I =====
    "I": HandshapeConfig(
        name="I",
        description="Pinky extended, other fingers in fist",
        pose=make_pose(
            index=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=THUMB_ALONGSIDE
        )
    ),
    
    # ===== J =====
    "J": HandshapeConfig(
        name="J",
        description="Same as I but with J motion (pinky traces J in air)",
        pose=make_pose(
            index=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=THUMB_ALONGSIDE
        ),
        movement_required=True
    ),
    
    # ===== K =====
    "K": HandshapeConfig(
        name="K",
        description="Index up, middle angled, thumb between them",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.3, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ),
    
    # ===== L =====
    "L": HandshapeConfig(
        name="L",
        description="Index up, thumb extended to side forming L shape",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_EXTENDED
        )
    ),
    
    # ===== M =====
    "M": HandshapeConfig(
        name="M",
        description="Fist with thumb under first three fingers",
        pose=make_pose(
            index=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            middle=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ring=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[1.2, 0.8, 0.0], [0.6, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
    ),
    
    # ===== N =====
    "N": HandshapeConfig(
        name="N",
        description="Fist with thumb under first two fingers",
        pose=make_pose(
            index=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            middle=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[1.0, 0.8, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
    ),
    
    # ===== O =====
    "O": HandshapeConfig(
        name="O",
        description="All fingertips touch thumb to form O shape",
        pose=make_pose(
            index=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            middle=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            ring=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            pinky=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            thumb=[[0.5, 0.4, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    # ===== P =====
    "P": HandshapeConfig(
        name="P",
        description="Like K but pointing down",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.3, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ),
    
    # ===== Q =====
    "Q": HandshapeConfig(
        name="Q",
        description="Like G but pointing down",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ),
    
    # ===== R =====
    "R": HandshapeConfig(
        name="R",
        description="Index and middle crossed, other fingers curled",
        pose=make_pose(
            index=[[0.0, 0.2, 0.0], EXTENDED, EXTENDED],  # Slight spread
            middle=[[0.0, -0.2, 0.0], EXTENDED, EXTENDED],  # Crosses over index
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== S =====
    "S": HandshapeConfig(
        name="S",
        description="Fist with thumb over curled fingers",
        pose=make_pose(
            index=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.8, 0.6, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    # ===== T =====
    "T": HandshapeConfig(
        name="T",
        description="Fist with thumb between index and middle finger",
        pose=make_pose(
            index=[[1.3, 0.0, 0.0], [0.8, 0.0, 0.0], [0.5, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=[[0.9, 0.7, 0.0], [0.4, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    # ===== U =====
    "U": HandshapeConfig(
        name="U",
        description="Index and middle extended together, other fingers curled",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== V =====
    "V": HandshapeConfig(
        name="V",
        description="Index and middle extended and spread apart (peace sign)",
        pose=make_pose(
            index=[[0.0, 0.3, 0.0], EXTENDED, EXTENDED],
            middle=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== W =====
    "W": HandshapeConfig(
        name="W",
        description="Index, middle, and ring extended and spread",
        pose=make_pose(
            index=[[0.0, 0.3, 0.0], EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    # ===== X =====
    "X": HandshapeConfig(
        name="X",
        description="Index bent at middle joint (hook shape), others curled",
        pose=make_pose(
            index=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_ALONGSIDE
        )
    ),
    
    # ===== Y =====
    "Y": HandshapeConfig(
        name="Y",
        description="Thumb and pinky extended, other fingers curled",
        pose=make_pose(
            index=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[EXTENDED, [0.0, 0.3, 0.0], EXTENDED],
            thumb=THUMB_EXTENDED
        )
    ),
    
    # ===== Z =====
    "Z": HandshapeConfig(
        name="Z",
        description="Index extended, traces Z in air",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_ALONGSIDE
        ),
        movement_required=True
    ),
    
    # ============================================================================
    # NUMBERS 0-9
    # ============================================================================
    
    "0": HandshapeConfig(
        name="0",
        description="Same as O - all fingertips touch thumb",
        pose=make_pose(
            index=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            middle=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            ring=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            pinky=[[0.7, 0.0, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]],
            thumb=[[0.5, 0.4, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    "1": HandshapeConfig(
        name="1",
        description="Index extended up, others curled",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_ALONGSIDE
        )
    ),
    
    "2": HandshapeConfig(
        name="2",
        description="Index and middle extended and spread (like V)",
        pose=make_pose(
            index=[[0.0, 0.3, 0.0], EXTENDED, EXTENDED],
            middle=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    "3": HandshapeConfig(
        name="3",
        description="Thumb, index, and middle extended",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "4": HandshapeConfig(
        name="4",
        description="Four fingers extended and spread, thumb across palm",
        pose=make_pose(
            index=[SPREAD_EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            thumb=THUMB_ACROSS_PALM
        )
    ),
    
    "5": HandshapeConfig(
        name="5",
        description="All five fingers extended and spread",
        pose=make_pose(
            index=[SPREAD_EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.0, 0.1, 0.0], EXTENDED, EXTENDED],
            ring=[[0.0, -0.1, 0.0], EXTENDED, EXTENDED],
            pinky=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "6": HandshapeConfig(
        name="6",
        description="Thumb and pinky touch, other three extended",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[[0.8, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]],
            thumb=[[0.6, 0.5, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    "7": HandshapeConfig(
        name="7",
        description="Thumb and ring touch, others extended",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[[0.8, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=[[0.6, 0.5, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    "8": HandshapeConfig(
        name="8",
        description="Thumb and middle touch, others extended",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.8, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=[[0.6, 0.5, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    "9": HandshapeConfig(
        name="9",
        description="Thumb and index touch (like F), others extended",
        pose=make_pose(
            index=[[0.8, 0.0, 0.0], [0.6, 0.0, 0.0], [0.4, 0.0, 0.0]],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=[[0.6, 0.5, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]]
        )
    ),
    
    # ============================================================================
    # COMMON ASL HANDSHAPES (Non-alphabet)
    # ============================================================================
    
    "ILY": HandshapeConfig(
        name="ILY",
        description="I Love You - thumb, index, and pinky extended",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "FLAT_O": HandshapeConfig(
        name="FLAT_O",
        description="Flattened O - fingertips bunched together",
        pose=make_pose(
            index=[[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
            middle=[[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
            ring=[[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
            pinky=[[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
            thumb=[[0.4, 0.3, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]
        )
    ),
    
    "CLAW": HandshapeConfig(
        name="CLAW",
        description="All fingers spread and curved like a claw",
        pose=make_pose(
            index=[[0.0, 0.3, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0]],
            middle=[[0.0, 0.1, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0]],
            ring=[[0.0, -0.1, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0]],
            pinky=[[0.0, -0.3, 0.0], [0.8, 0.0, 0.0], [0.6, 0.0, 0.0]],
            thumb=[[0.3, 0.3, 0.0], [0.5, 0.0, 0.0], [0.4, 0.0, 0.0]]
        )
    ),
    
    "BENT_V": HandshapeConfig(
        name="BENT_V",
        description="Index and middle bent at middle joint",
        pose=make_pose(
            index=[[0.0, 0.3, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            middle=[[0.0, -0.3, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_TUCKED
        )
    ),
    
    "OPEN_8": HandshapeConfig(
        name="OPEN_8",
        description="Middle finger bent, others extended (used in FEEL, SICK)",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ring=[EXTENDED, EXTENDED, EXTENDED],
            pinky=[EXTENDED, EXTENDED, EXTENDED],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "BABY_O": HandshapeConfig(
        name="BABY_O",
        description="Small O made with index and thumb, others relaxed",
        pose=make_pose(
            index=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            middle=[[0.3, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]],
            ring=[[0.3, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]],
            pinky=[[0.3, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]],
            thumb=[[0.5, 0.5, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]]
        )
    ),
    
    # Classifier handshapes (CL)
    "CL_1": HandshapeConfig(
        name="CL_1",
        description="Classifier 1 - index extended (person, thin object)",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_ALONGSIDE
        )
    ),
    
    "CL_3": HandshapeConfig(
        name="CL_3",
        description="Classifier 3 - thumb, index, middle (vehicle)",
        pose=make_pose(
            index=[EXTENDED, EXTENDED, EXTENDED],
            middle=[EXTENDED, EXTENDED, EXTENDED],
            ring=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            pinky=[FULLY_FLEXED, FULLY_FLEXED, [1.2, 0.0, 0.0]],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "CL_5": HandshapeConfig(
        name="CL_5",
        description="Classifier 5 - all fingers spread (large flat surface)",
        pose=make_pose(
            index=[SPREAD_EXTENDED, EXTENDED, EXTENDED],
            middle=[[0.0, 0.1, 0.0], EXTENDED, EXTENDED],
            ring=[[0.0, -0.1, 0.0], EXTENDED, EXTENDED],
            pinky=[[0.0, -0.3, 0.0], EXTENDED, EXTENDED],
            thumb=THUMB_EXTENDED
        )
    ),
    
    "CL_C": HandshapeConfig(
        name="CL_C",
        description="Classifier C - curved hand (cylindrical objects)",
        pose=make_pose(
            index=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            middle=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            ring=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            pinky=[[0.6, 0.0, 0.0], [0.4, 0.0, 0.0], [0.3, 0.0, 0.0]],
            thumb=[[0.4, 0.3, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]
        )
    ),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_handshape(name: str) -> HandshapeConfig:
    """Get a handshape configuration by name."""
    name_upper = name.upper()
    if name_upper not in ASL_HANDSHAPES:
        available = ", ".join(sorted(ASL_HANDSHAPES.keys()))
        raise ValueError(f"Unknown handshape: {name}. Available: {available}")
    return ASL_HANDSHAPES[name_upper]


def get_pose_array(name: str) -> np.ndarray:
    """Get handshape pose as numpy array (15, 3)."""
    config = get_handshape(name)
    return np.array(config.pose, dtype=np.float32)


def get_all_handshapes() -> List[str]:
    """Get list of all available handshape names."""
    return list(ASL_HANDSHAPES.keys())


def get_alphabet_handshapes() -> List[str]:
    """Get just the alphabet handshapes A-Z."""
    return [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def get_number_handshapes() -> List[str]:
    """Get number handshapes 0-9."""
    return [str(i) for i in range(10)]


def get_classifier_handshapes() -> List[str]:
    """Get classifier handshapes."""
    return [k for k in ASL_HANDSHAPES.keys() if k.startswith("CL_")]


def validate_pose(pose: List[List[float]]) -> bool:
    """Validate that a pose has correct structure."""
    if len(pose) != 15:
        return False
    for joint in pose:
        if len(joint) != 3:
            return False
        for angle in joint:
            if not isinstance(angle, (int, float)):
                return False
    return True


if __name__ == "__main__":
    # Test the handshapes
    print(f"Total handshapes defined: {len(ASL_HANDSHAPES)}")
    print(f"Alphabet: {get_alphabet_handshapes()}")
    print(f"Numbers: {get_number_handshapes()}")
    print(f"Classifiers: {get_classifier_handshapes()}")
    
    # Validate all poses
    for name, config in ASL_HANDSHAPES.items():
        if not validate_pose(config.pose):
            print(f"WARNING: Invalid pose for {name}")
        else:
            print(f"✓ {name}: {config.description[:50]}...")
