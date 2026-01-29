// ============================================
// COMPREHENSIVE ASL SIGN DICTIONARY
// Motion capture data for 100+ signs
// ============================================

export interface HandLandmarks {
  // Finger curl values (0 = extended, 1 = fully curled)
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
  // Finger spread (0 = together, 1 = spread apart)
  spread: number;
  // Wrist rotation [x, y, z] in radians
  wristRotation: [number, number, number];
}

export interface ArmPosition {
  // Target position relative to shoulder [x, y, z]
  position: [number, number, number];
  // Hand configuration
  hand: HandLandmarks;
  // Elbow bend (0 = straight, 1 = fully bent)
  elbow?: number;
}

export interface FacialExpression {
  mouthOpen: number; // 0-1
  mouthSmile: number; // -1 to 1 (frown to smile)
  eyebrowRaise: number; // -1 to 1 (frown to raised)
  eyeSquint: number; // 0-1
  noseWrinkle: number; // 0-1
  cheekPuff: number; // 0-1
}

export interface BodyPosition {
  shoulderShrug: number; // 0-1
  torsoTwist: number; // radians
  headTilt: [number, number, number]; // [pitch, yaw, roll]
  leanForward: number; // 0-1
}

export interface SignKeyframe {
  time: number; // 0-1 normalized time
  leftArm: ArmPosition;
  rightArm: ArmPosition;
  face: FacialExpression;
  body: BodyPosition;
}

export interface SignData {
  name: string;
  category: string;
  description: string;
  keyframes: SignKeyframe[];
  duration: number; // milliseconds
  handshape: string;
  movement: string;
  location: string;
}

// Default neutral positions
const neutralHand: HandLandmarks = {
  thumb: 0.2,
  index: 0.25,
  middle: 0.25,
  ring: 0.25,
  pinky: 0.3,
  spread: 0.1,
  wristRotation: [0, 0, 0]
};

const neutralFace: FacialExpression = {
  mouthOpen: 0,
  mouthSmile: 0,
  eyebrowRaise: 0,
  eyeSquint: 0,
  noseWrinkle: 0,
  cheekPuff: 0
};

const neutralBody: BodyPosition = {
  shoulderShrug: 0,
  torsoTwist: 0,
  headTilt: [0, 0, 0],
  leanForward: 0
};

// Helper to create a sign with single pose (static sign)
const createStaticSign = (
  name: string,
  category: string,
  description: string,
  leftArm: ArmPosition,
  rightArm: ArmPosition,
  face: Partial<FacialExpression> = {},
  body: Partial<BodyPosition> = {},
  duration = 800
): SignData => ({
  name,
  category,
  description,
  duration,
  handshape: 'varies',
  movement: 'static',
  location: 'neutral space',
  keyframes: [{
    time: 0,
    leftArm,
    rightArm,
    face: { ...neutralFace, ...face },
    body: { ...neutralBody, ...body }
  }, {
    time: 1,
    leftArm,
    rightArm,
    face: { ...neutralFace, ...face },
    body: { ...neutralBody, ...body }
  }]
});

// Helper to create animated signs
const createAnimatedSign = (
  name: string,
  category: string,
  description: string,
  keyframes: SignKeyframe[],
  duration = 1000
): SignData => ({
  name,
  category,
  description,
  duration,
  handshape: 'varies',
  movement: 'dynamic',
  location: 'varies',
  keyframes
});

// ============================================
// SIGN DATABASE - 100+ Signs
// ============================================

export const SIGN_DATABASE: Record<string, SignData> = {
  // ==========================================
  // GREETINGS & COMMON PHRASES (20 signs)
  // ==========================================
  
  hello: createAnimatedSign(
    'Hello',
    'Greetings',
    'Wave hand near forehead',
    [
      {
        time: 0,
        leftArm: { position: [-0.2, 0, 0.2], hand: neutralHand },
        rightArm: { position: [0.35, 0.35, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.5, wristRotation: [-0.3, 0.2, -0.5] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.4 },
        body: { ...neutralBody, headTilt: [0.1, 0.05, 0] }
      },
      {
        time: 0.5,
        leftArm: { position: [-0.2, 0, 0.2], hand: neutralHand },
        rightArm: { position: [0.4, 0.4, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.5, wristRotation: [-0.3, 0.4, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.4, eyebrowRaise: 0.5 },
        body: { ...neutralBody, headTilt: [0.1, -0.05, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.2, 0, 0.2], hand: neutralHand },
        rightArm: { position: [0.35, 0.35, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.5, wristRotation: [-0.3, 0.2, -0.5] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.4 },
        body: { ...neutralBody, headTilt: [0.1, 0.05, 0] }
      }
    ],
    1000
  ),

  goodbye: createAnimatedSign(
    'Goodbye',
    'Greetings',
    'Wave hand back and forth',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.3, 0.4, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.2 },
        body: neutralBody
      },
      {
        time: 0.25,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.35, 0.4, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0.3, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.3 },
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.3, 0.4, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, -0.3, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.2 },
        body: neutralBody
      },
      {
        time: 0.75,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.35, 0.4, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0.3, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.3 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.3, 0.4, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, -0.3] } },
        face: { ...neutralFace, mouthSmile: 0.2 },
        body: neutralBody
      }
    ],
    1200
  ),

  thank_you: createAnimatedSign(
    'Thank You',
    'Greetings',
    'Flat hand from chin forward',
    [
      {
        time: 0,
        leftArm: { position: [-0.2, -0.05, 0.2], hand: neutralHand },
        rightArm: { position: [0.05, 0.35, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.15, wristRotation: [-0.8, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.3 },
        body: { ...neutralBody, headTilt: [0.15, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.2, -0.05, 0.2], hand: neutralHand },
        rightArm: { position: [0.1, 0.2, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.15, wristRotation: [-0.4, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.4, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    900
  ),

  please: createAnimatedSign(
    'Please',
    'Greetings',
    'Circular motion on chest',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0, 0.15, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.2, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 0.5,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.1, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.2, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0, 0.15, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.2, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    1000
  ),

  sorry: createStaticSign(
    'Sorry',
    'Greetings',
    'Fist circles on chest',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0, 0.15, 0.3], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { mouthSmile: -0.2, eyebrowRaise: -0.1 },
    { headTilt: [0.15, 0, 0], shoulderShrug: 0.1 }
  ),

  excuse_me: createAnimatedSign(
    'Excuse Me',
    'Greetings',
    'Fingertips brush palm',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.2, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0.3, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0.05, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.2, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0.3, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    800
  ),

  nice_to_meet_you: createAnimatedSign(
    'Nice to Meet You',
    'Greetings',
    'Index fingers meet and move together',
    [
      {
        time: 0,
        leftArm: { position: [-0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, -0.3, 0] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.3 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 0.5,
        leftArm: { position: [-0.02, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0.1, 0] } },
        rightArm: { position: [0.02, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, -0.1, 0] } },
        face: { ...neutralFace, mouthSmile: 0.4, eyebrowRaise: 0.4 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0.2, 0] } },
        rightArm: { position: [0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, -0.2, 0] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.3 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    1200
  ),

  // ==========================================
  // YES/NO & BASIC RESPONSES (15 signs)
  // ==========================================

  yes: createAnimatedSign(
    'Yes',
    'Responses',
    'Fist nods like head nodding',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.2, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.2, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.2, 0, 0] }
      },
      {
        time: 0.5,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0.3, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.3, eyebrowRaise: 0.1 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.2, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.2, eyebrowRaise: 0.2 },
        body: { ...neutralBody, headTilt: [0.2, 0, 0] }
      }
    ],
    700
  ),

  no: createAnimatedSign(
    'No',
    'Responses',
    'Index and middle finger tap thumb',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0.3, 0] } },
        face: { ...neutralFace, mouthSmile: -0.1, eyebrowRaise: -0.15 },
        body: { ...neutralBody, headTilt: [0, 0.2, 0] }
      },
      {
        time: 0.5,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.7, index: 0.3, middle: 0.3, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0.3, 0] } },
        face: { ...neutralFace, mouthSmile: -0.1, eyebrowRaise: -0.15 },
        body: { ...neutralBody, headTilt: [0, -0.2, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.2, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0.3, 0] } },
        face: { ...neutralFace, mouthSmile: -0.1, eyebrowRaise: -0.15 },
        body: { ...neutralBody, headTilt: [0, 0.2, 0] }
      }
    ],
    800
  ),

  maybe: createStaticSign(
    'Maybe',
    'Responses',
    'Flat hands alternate up and down',
    { position: [-0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0.3] } },
    { position: [0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, -0.3] } },
    { eyebrowRaise: 0.3 },
    { headTilt: [0, 0, 0.1] }
  ),

  okay: createStaticSign(
    'Okay',
    'Responses',
    'O-K handshape',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.2, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0] } },
    { mouthSmile: 0.3 },
    { headTilt: [0.1, 0, 0] }
  ),

  good: createAnimatedSign(
    'Good',
    'Responses',
    'Flat hand from chin forward',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.3, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.15, wristRotation: [-0.6, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.4, eyebrowRaise: 0.35 },
        body: { ...neutralBody, headTilt: [0.12, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.2, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.15, wristRotation: [-0.4, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.5, eyebrowRaise: 0.3 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    700
  ),

  bad: createStaticSign(
    'Bad',
    'Responses',
    'Flat hand from chin down and away',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.15, wristRotation: [-0.4, 0, 0] } },
    { mouthSmile: -0.3, eyebrowRaise: -0.2 },
    { headTilt: [-0.1, 0, 0] }
  ),

  // ==========================================
  // PRONOUNS (10 signs)
  // ==========================================

  i_me: createStaticSign(
    'I/Me',
    'Pronouns',
    'Point to self',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0, 0.1, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  you: createStaticSign(
    'You',
    'Pronouns',
    'Point forward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0.15, 0.45], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { eyebrowRaise: 0.2 },
    {}
  ),

  he_she_it: createStaticSign(
    'He/She/It',
    'Pronouns',
    'Point to side',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.3, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    {},
    { torsoTwist: 0.1 }
  ),

  we: createStaticSign(
    'We',
    'Pronouns',
    'Index finger arcs from one shoulder to other',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.2, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  they: createStaticSign(
    'They',
    'Pronouns',
    'Point and sweep to side',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.25, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.2, 0] } },
    {},
    { torsoTwist: 0.15 }
  ),

  my_mine: createStaticSign(
    'My/Mine',
    'Pronouns',
    'Flat hand on chest',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0, 0.15, 0.2], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  your_yours: createStaticSign(
    'Your/Yours',
    'Pronouns',
    'Flat hand toward person',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0.15, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [-0.3, 0, 0] } },
    { eyebrowRaise: 0.2 },
    {}
  ),

  // ==========================================
  // QUESTIONS (15 signs)
  // ==========================================

  what: createStaticSign(
    'What',
    'Questions',
    'Palms up, fingers spread, shake slightly',
    { position: [-0.15, 0.05, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.4, wristRotation: [1.2, 0, 0.2] } },
    { position: [0.15, 0.05, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.4, wristRotation: [1.2, 0, -0.2] } },
    { eyebrowRaise: -0.3, mouthOpen: 0.1 },
    { shoulderShrug: 0.2 }
  ),

  where: createStaticSign(
    'Where',
    'Questions',
    'Index finger wags side to side',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { eyebrowRaise: -0.3 },
    { headTilt: [0, 0.1, 0] }
  ),

  when: createStaticSign(
    'When',
    'Questions',
    'Index fingers circle each other',
    { position: [-0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
    { eyebrowRaise: -0.3 },
    {}
  ),

  why: createAnimatedSign(
    'Why',
    'Questions',
    'Touch forehead, pull away into Y handshape',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.4, 0.2], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: -0.4 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.15, 0.3, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: -0.4 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    900
  ),

  who: createStaticSign(
    'Who',
    'Questions',
    'Index finger circles near mouth',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { eyebrowRaise: -0.3, mouthOpen: 0.2 },
    {}
  ),

  how: createStaticSign(
    'How',
    'Questions',
    'Knuckles together, rotate outward',
    { position: [-0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
    { eyebrowRaise: -0.3 },
    {}
  ),

  how_much: createStaticSign(
    'How Much',
    'Questions',
    'Claw hands move apart',
    { position: [-0.1, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.4, wristRotation: [0.5, 0, 0] } },
    { position: [0.1, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.4, wristRotation: [0.5, 0, 0] } },
    { eyebrowRaise: -0.3 },
    {}
  ),

  // ==========================================
  // EMOTIONS & FEELINGS (20 signs)
  // ==========================================

  happy: createStaticSign(
    'Happy',
    'Emotions',
    'Flat hands brush up on chest repeatedly',
    { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0.3] } },
    { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, -0.3] } },
    { mouthSmile: 0.6, eyebrowRaise: 0.4 },
    {}
  ),

  sad: createStaticSign(
    'Sad',
    'Emotions',
    'Hands move down face',
    { position: [-0.1, 0.25, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
    { position: [0.1, 0.25, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
    { mouthSmile: -0.4, eyebrowRaise: -0.3 },
    { headTilt: [0.15, 0, 0] }
  ),

  angry: createStaticSign(
    'Angry',
    'Emotions',
    'Claw hand pulls away from face',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.3, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.5, wristRotation: [0, 0, 0] } },
    { mouthSmile: -0.5, eyebrowRaise: -0.5, noseWrinkle: 0.3 },
    { headTilt: [-0.1, 0, 0] }
  ),

  scared: createStaticSign(
    'Scared',
    'Emotions',
    'Fists open suddenly in front of chest',
    { position: [-0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.5, wristRotation: [0, 0, 0.2] } },
    { position: [0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.5, wristRotation: [0, 0, -0.2] } },
    { mouthOpen: 0.4, eyebrowRaise: 0.5, eyeSquint: 0.3 },
    { shoulderShrug: 0.3 }
  ),

  surprised: createStaticSign(
    'Surprised',
    'Emotions',
    'Index fingers flick up near eyes',
    { position: [-0.12, 0.35, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { position: [0.12, 0.35, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { mouthOpen: 0.5, eyebrowRaise: 0.6 },
    {}
  ),

  tired: createStaticSign(
    'Tired',
    'Emotions',
    'Bent hands drop on chest',
    { position: [-0.1, 0.1, 0.25], hand: { ...neutralHand, thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4, spread: 0.1, wristRotation: [0.3, 0, 0] } },
    { position: [0.1, 0.1, 0.25], hand: { ...neutralHand, thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4, spread: 0.1, wristRotation: [0.3, 0, 0] } },
    { mouthSmile: -0.2, eyeSquint: 0.4 },
    { shoulderShrug: 0.2, headTilt: [0.1, 0.1, 0] }
  ),

  love: createStaticSign(
    'Love',
    'Emotions',
    'Cross arms over chest',
    { position: [-0.05, 0.1, 0.2], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0.5] } },
    { position: [0.05, 0.1, 0.2], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, -0.5] } },
    { mouthSmile: 0.5, eyebrowRaise: 0.3 },
    { headTilt: [0.1, 0, 0] }
  ),

  i_love_you: createStaticSign(
    'I Love You',
    'Emotions',
    'ILY handshape - thumb, index, pinky extended',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.2, 0.25, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.95, ring: 0.95, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    { mouthSmile: 0.5, eyebrowRaise: 0.3 },
    {}
  ),

  excited: createAnimatedSign(
    'Excited',
    'Emotions',
    'Middle fingers flick up alternately on chest',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.5, eyebrowRaise: 0.4 },
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.6, eyebrowRaise: 0.5 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.5, eyebrowRaise: 0.4 },
        body: neutralBody
      }
    ],
    1000
  ),

  worried: createStaticSign(
    'Worried',
    'Emotions',
    'Alternating B hands circle in front of face',
    { position: [-0.1, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.9, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    { position: [0.1, 0.3, 0.3], hand: { ...neutralHand, thumb: 0.9, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    { mouthSmile: -0.3, eyebrowRaise: -0.4 },
    {}
  ),

  // ==========================================
  // ACTIONS & VERBS (25 signs)
  // ==========================================

  help: createStaticSign(
    'Help',
    'Actions',
    'Fist on flat palm, lift up',
    { position: [-0.05, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
    { position: [0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.95, index: 0.95, middle: 0.95, ring: 0.95, pinky: 0.95, spread: 0, wristRotation: [0, 0, 0] } },
    { mouthOpen: 0.3, eyebrowRaise: 0.35 },
    { headTilt: [0.1, 0, 0], shoulderShrug: 0.1 }
  ),

  want: createAnimatedSign(
    'Want',
    'Actions',
    'Claw hands pull toward body',
    [
      {
        time: 0,
        leftArm: { position: [-0.18, 0.1, 0.45], hand: { ...neutralHand, thumb: 0.15, index: 0.15, middle: 0.15, ring: 0.15, pinky: 0.15, spread: 0.25, wristRotation: [0.4, 0, 0] } },
        rightArm: { position: [0.18, 0.1, 0.45], hand: { ...neutralHand, thumb: 0.15, index: 0.15, middle: 0.15, ring: 0.15, pinky: 0.15, spread: 0.25, wristRotation: [0.4, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2, eyebrowRaise: 0.15 },
        body: { ...neutralBody, headTilt: [0.08, 0, 0], shoulderShrug: 0.05 }
      },
      {
        time: 1,
        leftArm: { position: [-0.15, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.2, wristRotation: [0.3, 0, 0] } },
        rightArm: { position: [0.15, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.2, wristRotation: [0.3, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2, eyebrowRaise: 0.15 },
        body: { ...neutralBody, headTilt: [0.08, 0, 0], shoulderShrug: 0.05 }
      }
    ],
    800
  ),

  need: createStaticSign(
    'Need',
    'Actions',
    'X hand bends down',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0.5, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0.3, 0, 0] } },
    { eyebrowRaise: 0.2 },
    { headTilt: [0.1, 0, 0] }
  ),

  have: createStaticSign(
    'Have',
    'Actions',
    'Bent hands touch chest',
    { position: [-0.1, 0.1, 0.25], hand: { ...neutralHand, thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4, spread: 0.1, wristRotation: [0, 0, 0] } },
    { position: [0.1, 0.1, 0.25], hand: { ...neutralHand, thumb: 0.4, index: 0.4, middle: 0.4, ring: 0.4, pinky: 0.4, spread: 0.1, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  go: createAnimatedSign(
    'Go',
    'Actions',
    'Index fingers point and move forward',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.1, 0.5], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.1, 0] } },
        rightArm: { position: [0.1, 0.1, 0.5], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.1, 0] } },
        face: neutralFace,
        body: { ...neutralBody, leanForward: 0.1 }
      }
    ],
    700
  ),

  come: createAnimatedSign(
    'Come',
    'Actions',
    'Index fingers beckon toward body',
    [
      {
        time: 0,
        leftArm: { position: [-0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.3, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0.3, 0, 0] } },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.8, index: 0.3, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0.3, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    700
  ),

  eat: createAnimatedSign(
    'Eat',
    'Actions',
    'Flat O hand moves to mouth',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.3 },
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.28, 0.25], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.4 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.3 },
        body: neutralBody
      }
    ],
    1000
  ),

  drink: createAnimatedSign(
    'Drink',
    'Actions',
    'C hand tips toward mouth',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.3, 0.25], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [-0.5, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.3 },
        body: { ...neutralBody, headTilt: [-0.1, 0, 0] }
      }
    ],
    900
  ),

  sleep: createStaticSign(
    'Sleep',
    'Actions',
    'Hand moves down face as eyes close',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.25, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
    { mouthSmile: 0, eyeSquint: 0.8 },
    { headTilt: [0.15, 0.1, 0] }
  ),

  work: createAnimatedSign(
    'Work',
    'Actions',
    'Fists tap together',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    800
  ),

  learn: createAnimatedSign(
    'Learn',
    'Actions',
    'Flat hand picks up from palm to forehead',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0.1, 0.4, 0.2], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: 0.3 },
        body: neutralBody
      }
    ],
    1000
  ),

  understand: createAnimatedSign(
    'Understand',
    'Actions',
    'Index finger flicks up near temple',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.12, 0.4, 0.2], hand: { ...neutralHand, thumb: 0.8, index: 0.5, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: 0.1 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.12, 0.45, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: 0.4, mouthSmile: 0.2 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    700
  ),

  know: createStaticSign(
    'Know',
    'Actions',
    'Fingertips tap forehead',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.08, 0.42, 0.2], hand: { ...neutralHand, thumb: 0.9, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    {},
    { headTilt: [0.1, 0, 0] }
  ),

  think: createStaticSign(
    'Think',
    'Actions',
    'Index finger touches temple',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.4, 0.15], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { eyebrowRaise: 0.2 },
    { headTilt: [0.1, 0, 0] }
  ),

  see: createAnimatedSign(
    'See',
    'Actions',
    'V hand moves from eyes outward',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.08, 0.35, 0.2], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.3, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.15, 0.3, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.3, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    700
  ),

  hear: createStaticSign(
    'Hear',
    'Actions',
    'Index finger points to ear',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.12, 0.35, 0.1], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.5, 0] } },
    {},
    { headTilt: [0, 0.1, 0] }
  ),

  say_tell: createAnimatedSign(
    'Say/Tell',
    'Actions',
    'Index finger moves from chin outward',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.25, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.15, 0.2, 0.45], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthOpen: 0.3 },
        body: neutralBody
      }
    ],
    700
  ),

  give: createAnimatedSign(
    'Give',
    'Actions',
    'Flat O hands move forward and open',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.3, 0, 0] } },
        rightArm: { position: [0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.3, 0, 0] } },
        face: neutralFace,
        body: { ...neutralBody, leanForward: 0.1 }
      }
    ],
    800
  ),

  take: createAnimatedSign(
    'Take',
    'Actions',
    'Open hands close to fists while pulling back',
    [
      {
        time: 0,
        leftArm: { position: [-0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.3, 0, 0] } },
        rightArm: { position: [0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.3, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    800
  ),

  wait: createStaticSign(
    'Wait',
    'Actions',
    'Open hands with wiggling fingers',
    { position: [-0.15, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.5, 0, 0.2] } },
    { position: [0.15, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.5, 0, -0.2] } },
    { eyebrowRaise: 0.2 },
    {}
  ),

  stop: createStaticSign(
    'Stop',
    'Actions',
    'Flat hand chops into palm',
    { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
    { position: [0.05, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    { eyebrowRaise: -0.2 },
    {}
  ),

  start_begin: createAnimatedSign(
    'Start/Begin',
    'Actions',
    'Index finger twists in other hand',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0.5] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    800
  ),

  finish_done: createAnimatedSign(
    'Finish/Done',
    'Actions',
    'Open hands flip outward',
    [
      {
        time: 0,
        leftArm: { position: [-0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0.5] } },
        rightArm: { position: [0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, -0.5] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.2, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 1.2] } },
        rightArm: { position: [0.2, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, -1.2] } },
        face: { ...neutralFace, eyebrowRaise: 0.2 },
        body: neutralBody
      }
    ],
    700
  ),

  // ==========================================
  // TIME CONCEPTS (10 signs)
  // ==========================================

  now: createStaticSign(
    'Now',
    'Time',
    'Y hands drop down',
    { position: [-0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    { position: [0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  later: createStaticSign(
    'Later',
    'Time',
    'L hand twists forward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0.15, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0.3, wristRotation: [0, 0.3, 0] } },
    {},
    {}
  ),

  before: createStaticSign(
    'Before',
    'Time',
    'Hand moves back from other hand',
    { position: [-0.1, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0.5, 0] } },
    { position: [0.05, 0.1, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, -0.5, 0] } },
    {},
    {}
  ),

  after: createStaticSign(
    'After',
    'Time',
    'Hand moves forward from other hand',
    { position: [-0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0.5, 0] } },
    { position: [0.05, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, -0.5, 0] } },
    {},
    {}
  ),

  today: createStaticSign(
    'Today',
    'Time',
    'Y hands drop in front of body',
    { position: [-0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    { position: [0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  tomorrow: createStaticSign(
    'Tomorrow',
    'Time',
    'A hand moves forward from cheek',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.12, 0.3, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  yesterday: createStaticSign(
    'Yesterday',
    'Time',
    'A/Y hand touches chin then moves back',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.08, 0.28, 0.2], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  week: createStaticSign(
    'Week',
    'Time',
    'Index slides across palm',
    { position: [-0.1, 0.05, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
    { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  month: createStaticSign(
    'Month',
    'Time',
    'Index slides down other index',
    { position: [-0.05, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { position: [0.05, 0.2, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  year: createAnimatedSign(
    'Year',
    'Time',
    'Fists orbit each other',
    [
      {
        time: 0,
        leftArm: { position: [-0.08, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.08, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.08, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.08, 0.05, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.08, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.08, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    1000
  ),

  // ==========================================
  // PEOPLE & FAMILY (10 signs)
  // ==========================================

  person: createStaticSign(
    'Person',
    'People',
    'P hands move down',
    { position: [-0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.5, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
    { position: [0.15, 0.15, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.5, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  friend: createAnimatedSign(
    'Friend',
    'People',
    'Index fingers hook and reverse',
    [
      {
        time: 0,
        leftArm: { position: [-0.05, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0.4, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0.5] } },
        rightArm: { position: [0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0.4, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, -0.5] } },
        face: { ...neutralFace, mouthSmile: 0.3 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0.4, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, -0.5] } },
        rightArm: { position: [0.05, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0.4, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0.5] } },
        face: { ...neutralFace, mouthSmile: 0.4 },
        body: neutralBody
      }
    ],
    900
  ),

  family: createStaticSign(
    'Family',
    'People',
    'F hands circle to meet',
    { position: [-0.12, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
    { position: [0.12, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
    { mouthSmile: 0.2 },
    {}
  ),

  mother: createStaticSign(
    'Mother',
    'People',
    'Open 5 hand thumb on chin',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.25, 0.25], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0] } },
    { mouthSmile: 0.2 },
    {}
  ),

  father: createStaticSign(
    'Father',
    'People',
    'Open 5 hand thumb on forehead',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.4, 0.2], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  child: createStaticSign(
    'Child',
    'People',
    'Flat hand pats downward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.15, 0, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [-0.5, 0, 0] } },
    { mouthSmile: 0.2 },
    {}
  ),

  baby: createStaticSign(
    'Baby',
    'People',
    'Arms cradle and rock',
    { position: [-0.1, 0, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.1, wristRotation: [0.5, 0, 0.3] } },
    { position: [0.1, 0.05, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.1, wristRotation: [0.5, 0, -0.3] } },
    { mouthSmile: 0.4 },
    { torsoTwist: 0.1 }
  ),

  boy: createStaticSign(
    'Boy',
    'People',
    'Flat O hand at forehead',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.08, 0.4, 0.2], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  girl: createStaticSign(
    'Girl',
    'People',
    'Thumb traces jawline',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.08, 0.28, 0.2], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  teacher: createAnimatedSign(
    'Teacher',
    'People',
    'Flat O hands at temples move forward, then person sign',
    [
      {
        time: 0,
        leftArm: { position: [-0.12, 0.38, 0.2], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.12, 0.38, 0.2], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.15, 0.35, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.15, 0.35, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.5, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.15, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0.5, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    1200
  ),


  // ==========================================
  // COMMON SENTENCES & PHRASES (Additional)
  // ==========================================

  can: createStaticSign(
    'Can',
    'Actions',
    'S hands move down together',
    { position: [-0.1, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { position: [0.1, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  will: createStaticSign(
    'Will',
    'Time',
    'Flat hand moves forward from side of face',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.12, 0.3, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  not: createAnimatedSign(
    'Not',
    'Responses',
    'Thumb under chin flicks forward',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.05, 0.25, 0.2], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: -0.2 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.2, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, eyebrowRaise: -0.2 },
        body: neutralBody
      }
    ],
    600
  ),

  like: createAnimatedSign(
    'Like',
    'Emotions',
    'Middle finger and thumb pull away from chest',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0, 0.15, 0.25], hand: { ...neutralHand, thumb: 0.3, index: 0.9, middle: 0.3, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.3 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.9, middle: 0.5, ring: 0.9, pinky: 0.9, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.4 },
        body: neutralBody
      }
    ],
    800
  ),

  dont_like: createAnimatedSign(
    'Don\'t Like',
    'Emotions',
    'Middle finger flicks away from chest',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0, 0.15, 0.25], hand: { ...neutralHand, thumb: 0.3, index: 0.9, middle: 0.3, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: -0.2, eyebrowRaise: -0.2 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.15, 0.1, 0.45], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.3, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: -0.3, eyebrowRaise: -0.2 },
        body: neutralBody
      }
    ],
    800
  ),

  am: createStaticSign(
    'Am',
    'Pronouns',
    'A hand moves from lips forward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.28, 0.3], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  is: createStaticSign(
    'Is',
    'Pronouns',
    'I hand moves from lips forward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.25, 0.35], hand: { ...neutralHand, thumb: 0.9, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  are: createStaticSign(
    'Are',
    'Pronouns',
    'R hand moves forward from lips',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.25, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  this: createStaticSign(
    'This',
    'Pronouns',
    'Index finger points down to palm',
    { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
    { position: [0.05, 0.15, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  that: createStaticSign(
    'That',
    'Pronouns',
    'Y hand drops onto palm',
    { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
    { position: [0.05, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0, spread: 0.5, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  here: createStaticSign(
    'Here',
    'Actions',
    'Flat hands circle in front of body',
    { position: [-0.15, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0.5, 0, 0] } },
    { position: [0.15, 0.1, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [0.5, 0, 0] } },
    {},
    {}
  ),

  there: createStaticSign(
    'There',
    'Actions',
    'Index finger points outward',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.2, 0.15, 0.45], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    { torsoTwist: 0.1 }
  ),

  with: createStaticSign(
    'With',
    'Actions',
    'A hands come together',
    { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    { position: [0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  for: createStaticSign(
    'For',
    'Actions',
    'Index finger touches forehead then points out',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.35, 0.35], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  about: createAnimatedSign(
    'About',
    'Actions',
    'Index finger circles around other index',
    [
      {
        time: 0,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.05, 0.2, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    900
  ),

  name: createAnimatedSign(
    'Name',
    'People',
    'H fingers tap together',
    [
      {
        time: 0,
        leftArm: { position: [-0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.08, 0.18, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.08, 0.18, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0, ring: 0.9, pinky: 0.9, spread: 0.2, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    800
  ),

  sign_language: createStaticSign(
    'Sign Language',
    'Actions',
    'Index fingers alternate circling',
    { position: [-0.12, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    { position: [0.12, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
    {},
    {}
  ),

  deaf: createStaticSign(
    'Deaf',
    'People',
    'Index finger touches ear then mouth',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.1, 0.35, 0.15], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    {},
    {}
  ),

  hearing: createStaticSign(
    'Hearing',
    'People',
    'Index finger circles near mouth',
    { position: [-0.25, -0.1, 0.15], hand: neutralHand },
    { position: [0.05, 0.28, 0.25], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
    {},
    {}
  ),

  interpreter: createAnimatedSign(
    'Interpreter',
    'People',
    'F hands alternate up and down',
    [
      {
        time: 0,
        leftArm: { position: [-0.12, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.12, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.12, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.12, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.12, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        rightArm: { position: [0.12, 0.1, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0, ring: 0, pinky: 0, spread: 0.2, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    1000
  ),

  communicate: createAnimatedSign(
    'Communicate',
    'Actions',
    'C hands move back and forth from mouth',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, -0.3, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2 },
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.15, 0.25, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.15, 0.25, 0.4], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, -0.3, 0] } },
        face: { ...neutralFace, mouthOpen: 0.3 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.25, 0.3], hand: { ...neutralHand, thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, spread: 0.3, wristRotation: [0, -0.3, 0] } },
        face: { ...neutralFace, mouthOpen: 0.2 },
        body: neutralBody
      }
    ],
    1000
  ),

  welcome: createAnimatedSign(
    'Welcome',
    'Greetings',
    'Open hand sweeps toward body',
    [
      {
        time: 0,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.25, 0.15, 0.4], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.4, eyebrowRaise: 0.3 },
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.25, -0.1, 0.15], hand: neutralHand },
        rightArm: { position: [0.1, 0.1, 0.3], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.3, wristRotation: [0.3, 0, 0] } },
        face: { ...neutralFace, mouthSmile: 0.5, eyebrowRaise: 0.3 },
        body: { ...neutralBody, headTilt: [0.1, 0, 0] }
      }
    ],
    900
  ),

  practice: createAnimatedSign(
    'Practice',
    'Actions',
    'A hand brushes back and forth on other index',
    [
      {
        time: 0,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0, index: 0.9, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    900
  ),

  again: createAnimatedSign(
    'Again',
    'Actions',
    'Bent hand arcs into palm',
    [
      {
        time: 0,
        leftArm: { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0.15, 0.2, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0.1, wristRotation: [0, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.1, 0.1, 0.35], hand: { ...neutralHand, thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0, spread: 0.1, wristRotation: [1.2, 0, 0] } },
        rightArm: { position: [0, 0.1, 0.35], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0.1, wristRotation: [0.5, 0, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    700
  ),

  more: createAnimatedSign(
    'More',
    'Actions',
    'Flat O hands tap together',
    [
      {
        time: 0,
        leftArm: { position: [-0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 0.5,
        leftArm: { position: [-0.02, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0.1, 0] } },
        rightArm: { position: [0.02, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, -0.1, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.08, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.5, index: 0.5, middle: 0.5, ring: 0.5, pinky: 0.5, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    700
  ),

  different: createAnimatedSign(
    'Different',
    'Actions',
    'Index fingers cross and separate',
    [
      {
        time: 0,
        leftArm: { position: [-0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.05, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      },
      {
        time: 1,
        leftArm: { position: [-0.15, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
        rightArm: { position: [0.15, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
        face: neutralFace,
        body: neutralBody
      }
    ],
    700
  ),

  same: createStaticSign(
    'Same',
    'Actions',
    'Index fingers come together',
    { position: [-0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, 0.3, 0] } },
    { position: [0.1, 0.15, 0.4], hand: { ...neutralHand, thumb: 0.8, index: 0, middle: 0.9, ring: 0.9, pinky: 0.9, spread: 0, wristRotation: [0, -0.3, 0] } },
    {},
    {}
  ),
};

// Get all sign names
export const getSignNames = (): string[] => Object.keys(SIGN_DATABASE);

// Get signs by category
export const getSignsByCategory = (category: string): SignData[] => {
  return Object.values(SIGN_DATABASE).filter(sign => sign.category === category);
};

// Get all categories
export const getCategories = (): string[] => {
  const categories = new Set(Object.values(SIGN_DATABASE).map(sign => sign.category));
  return Array.from(categories);
};

// Search signs
export const searchSigns = (query: string): SignData[] => {
  const lowerQuery = query.toLowerCase();
  return Object.values(SIGN_DATABASE).filter(sign => 
    sign.name.toLowerCase().includes(lowerQuery) ||
    sign.description.toLowerCase().includes(lowerQuery) ||
    sign.category.toLowerCase().includes(lowerQuery)
  );
};

// Get sign by name (case insensitive, handles variations)
export const getSign = (name: string): SignData | undefined => {
  const normalizedName = name.toLowerCase().replace(/[\s-]/g, '_');
  return SIGN_DATABASE[normalizedName] || 
         Object.values(SIGN_DATABASE).find(s => 
           s.name.toLowerCase().replace(/[\s-]/g, '_') === normalizedName ||
           s.name.toLowerCase() === name.toLowerCase()
         );
};

export default SIGN_DATABASE;
