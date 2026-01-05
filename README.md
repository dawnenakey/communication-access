# SLR Blender Synthetic Data Pipeline

**Author:** Dawnena Key / SonZo AI  
**License:** Proprietary - Patent Pending

Generate synthetic ASL handshape training data using Blender to augment your Sign Language Recognition (SLR) model training.

---

## Files Overview

| File | Purpose |
|------|---------|
| `asl_handshapes.py` | ASL handshape definitions (A-Z, 0-9, classifiers) as MANO pose parameters |
| `generate_synthetic_asl.py` | Blender script to generate synthetic hand images |
| `dataset_loader.py` | PyTorch dataset loader for training integration |
| `README.md` | This file |

---

## Quick Start

### 1. Install Blender

```bash
# Ubuntu/Debian
sudo apt-get install blender

# Or download from blender.org (recommended for latest version)
wget https://download.blender.org/release/Blender3.6/blender-3.6.5-linux-x64.tar.xz
tar -xf blender-3.6.5-linux-x64.tar.xz
export BLENDER_PATH=/path/to/blender-3.6.5-linux-x64/blender
```

### 2. Generate Synthetic Data

```bash
# Basic usage - generates all handshapes
blender --background --python generate_synthetic_asl.py

# Custom output directory and sample count
blender --background --python generate_synthetic_asl.py -- \
    --output /path/to/output \
    --samples 200

# Generate specific handshapes only
blender --background --python generate_synthetic_asl.py -- \
    --handshapes A,B,C,ILY,1,2,3 \
    --samples 100

# Use GPU rendering (faster)
blender --background --python generate_synthetic_asl.py -- \
    --gpu \
    --engine CYCLES
```

### 3. Load Data for Training

```python
from dataset_loader import SyntheticASLDataset, create_dataloaders

# Simple usage
train_loader, val_loader = create_dataloaders(
    synthetic_dir="/path/to/synthetic_data",
    real_dir="/path/to/real_signcut_data",  # optional
    batch_size=32,
    synthetic_ratio=0.5
)

# Train your model
for batch in train_loader:
    images = batch['image']       # (B, 3, 224, 224)
    labels = batch['label']       # (B,)
    keypoints = batch.get('keypoints_2d')  # (B, 16, 3) if available
    # ... training loop
```

---

## Configuration Options

### Generation Script (`generate_synthetic_asl.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--output` | `/tmp/synthetic_asl_data` | Output directory |
| `--samples` | 100 | Samples per handshape |
| `--handshapes` | All | Comma-separated list (e.g., `A,B,C`) |
| `--format` | PNG | Export format: `PNG` or `BASE64` |
| `--gpu` | False | Use GPU rendering |
| `--engine` | CYCLES | Render engine: `CYCLES` or `EEVEE` |
| `--seed` | None | Random seed for reproducibility |

### Dataset Loader

```python
# Full customization
dataset = SyntheticASLDataset(
    data_dir="/path/to/data",
    transform=custom_transforms,
    include_keypoints=True,
    target_handshapes=['A', 'B', 'C'],  # Filter
    max_samples_per_class=500           # Limit
)
```

---

## Curriculum Learning

The dataset supports curriculum learning with three phases:

```python
from dataset_loader import CombinedASLDataset

# Phase 1: Synthetic only (warm-up)
combined = CombinedASLDataset(
    synthetic_dataset, real_dataset,
    curriculum_phase="synthetic_only"
)
# Train for N epochs...

# Phase 2: Mixed training
combined.set_curriculum_phase("mixed")
# Train for M epochs...

# Phase 3: Real-heavy fine-tuning
combined.set_curriculum_phase("real_heavy")
# Train for K epochs...
```

---

## Output Structure

```
output_directory/
├── metadata.json      # All sample metadata
├── label_map.json     # Handshape -> integer mapping
├── A/
│   ├── A_0000.png
│   ├── A_0001.png
│   └── ...
├── B/
│   └── ...
└── ...
```

### Metadata Format

```json
{
  "samples": [
    {
      "id": 0,
      "handshape": "A",
      "skin_tone_idx": 2,
      "seed": 0,
      "filepath": "A/A_0000.png",
      "keypoints_2d": [[x, y, visibility], ...],
      "keypoints_3d": [[x, y, z], ...]
    }
  ]
}
```

---

## Handshapes Included

### Alphabet (26)
A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Numbers (10)
0, 1, 2, 3, 4, 5, 6, 7, 8, 9

### Common Signs (10)
- ILY (I Love You)
- FLAT_O
- CLAW
- BENT_V
- OPEN_8
- BABY_O
- CL_1 (Classifier 1 - person/thin object)
- CL_3 (Classifier 3 - vehicle)
- CL_5 (Classifier 5 - large surface)
- CL_C (Classifier C - cylindrical)

---

## Upgrading to MANO Hand Model

For photorealistic hands, replace the simple mesh with MANO:

1. **Register** at https://mano.is.tue.mpg.de/
2. **Download** `MANO_RIGHT.pkl` and `MANO_LEFT.pkl`
3. **Install** the MANO Blender add-on
4. **Modify** `generate_synthetic_asl.py`:

```python
def load_mano_model(mano_path, hand_type='right'):
    import pickle
    with open(mano_path, 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    # ... MANO mesh creation code
```

---

## Integration with Your 3D CNN + LSTM

Your model expects sequences of frames. To generate video sequences:

```python
# Modify CONFIG in generate_synthetic_asl.py
CONFIG.frames_per_sample = 16  # For sequence models
CONFIG.frame_interval_ms = 33  # ~30 fps

# The generator will output:
# output_dir/A/A_0000_frame00.png
# output_dir/A/A_0000_frame01.png
# ...
```

Or load as sequences in the dataset loader:

```python
class SequenceASLDataset(Dataset):
    def __getitem__(self, idx):
        # Load N consecutive frames
        frames = [self.load_frame(idx, f) for f in range(self.seq_len)]
        return torch.stack(frames), label
```

---

## Expected Results

Based on research:
- **+5% accuracy** improvement with synthetic augmentation
- **Better generalization** to unseen signers
- **Improved handling** of occlusions and lighting variations

---

## Troubleshooting

### Blender not found
```bash
export PATH=$PATH:/path/to/blender
```

### GPU rendering fails
```bash
# Fall back to CPU
blender --background --python generate_synthetic_asl.py -- --engine EEVEE
```

### Import errors in Blender
```bash
# Install packages to Blender's Python
/path/to/blender/3.6/python/bin/python3.10 -m pip install numpy
```

---

## Next Steps with Claude Code

1. Copy this folder to your SLR project
2. Open Claude Code in your project directory
3. Say: "Help me integrate this Blender synthetic data pipeline with my existing 3D CNN + LSTM model"
4. Claude Code will adapt the loader to your exact input format

---

## Contact

**Dawnena Key**  
SonZo AI - Founder/Chief AI Officer  
dawnena@sonzo.io
