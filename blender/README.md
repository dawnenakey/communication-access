# Blender MANO Rendering Pipeline

Generate synthetic ASL handshape training data using Blender and the MANO hand model.

## Overview

This pipeline generates realistic images of ASL handshapes for training sign language recognition models. It uses:
- **MANO pose parameters** from `asl_handshapes.py` (46 handshapes)
- **Blender** for photorealistic rendering
- **Randomized variations** (camera angles, lighting, skin tones, backgrounds)

## Directory Structure

```
blender/
├── mano_renderer.py    # Main Blender rendering script
├── batch_render.py     # Orchestrator for parallel batch rendering
├── test_pipeline.py    # Test suite for pipeline components
└── README.md           # This file
```

## Requirements

- **Blender 3.0+** (with Python API)
- **Python 3.10+**
- **MANO hand model** (optional, creates simple hand for testing)

### Install Blender

```bash
# Ubuntu/Debian
sudo apt install blender

# macOS (via Homebrew)
brew install --cask blender

# Or download from https://www.blender.org/download/
```

## Quick Start

### 1. Test the Pipeline

```bash
cd communication-access/blender
python test_pipeline.py
```

### 2. Test Render (Single Handshape)

```bash
# Run test with 3 samples of handshape "A"
blender --background --python mano_renderer.py -- --test

# Or using batch_render.py
python batch_render.py --test
```

### 3. Render All Handshapes

```bash
# Using batch_render.py (recommended)
python batch_render.py \
    --output ./synthetic_data \
    --samples 100 \
    --parallel 4 \
    --size 512

# Direct Blender command
blender --background --python mano_renderer.py -- \
    --all \
    --output ./synthetic_data \
    --samples 100
```

## Command Line Options

### mano_renderer.py (run inside Blender)

```bash
blender --background --python mano_renderer.py -- [OPTIONS]

Options:
  --handshape NAME    Render specific handshape (e.g., A, B, ILY)
  --all               Render all defined handshapes
  --output DIR        Output directory (default: ./synthetic_data)
  --model PATH        Path to MANO model file (.fbx, .obj, .blend)
  --samples N         Samples per handshape (default: 10)
  --size N            Image size in pixels (default: 512)
  --test              Run quick test render
```

### batch_render.py (orchestrator)

```bash
python batch_render.py [OPTIONS]

Options:
  -o, --output DIR        Output directory
  -s, --samples N         Samples per handshape (default: 100)
  -p, --parallel N        Parallel Blender processes (default: 1)
  -m, --model PATH        MANO model file
  --size N                Image size (default: 512)
  --alphabet-only         Only render A-Z
  --handshapes A B C      Specific handshapes to render
  --test                  Quick test with handshape A
```

## Output Format

```
synthetic_data/
├── A/
│   ├── A_0000.png
│   ├── A_0001.png
│   └── ...
├── B/
│   └── ...
├── metadata.json         # Combined sample metadata
├── label_map.json        # Handshape to label mapping
└── batch_summary.json    # Rendering statistics
```

### metadata.json Format

```json
{
  "version": "1.0",
  "total_samples": 4600,
  "samples": [
    {
      "id": 0,
      "sample_id": "A_0000",
      "handshape": "A",
      "filepath": "A/A_0000.png",
      "skin_tone_idx": 2,
      "background_type": "solid",
      "keypoints_2d": [[x, y], ...],
      "keypoints_3d": [[x, y, z], ...]
    }
  ]
}
```

## Using with Dataset Loader

```python
from dataset_loader import SyntheticASLDataset, create_dataloaders

# Load synthetic data
train_loader, val_loader = create_dataloaders(
    synthetic_dir="./synthetic_data",
    batch_size=32,
    image_size=224
)

# Iterate over batches
for batch in train_loader:
    images = batch['image']       # [B, 3, 224, 224]
    labels = batch['label']       # [B]
    handshapes = batch['handshape']  # List of strings
    keypoints = batch.get('keypoints_2d')  # Optional
```

## Using Your Own MANO Model

If you have a MANO model file:

```bash
# Support formats: .fbx, .obj, .blend
python batch_render.py --model /path/to/mano_hand.fbx
```

### Expected Bone Names

The renderer supports multiple bone naming conventions:

| Convention | Example Bone Names |
|------------|-------------------|
| Default | `index_01`, `index_02`, `thumb_01` |
| Rigify | `f_index.01.L`, `thumb.01.L` |
| MakeHuman | `index1_L`, `thumb1_L` |
| MANO | `index1`, `thumb1` |

The renderer auto-detects the naming convention.

## Customization

### Modify Render Settings

Edit `RenderConfig` class in `mano_renderer.py`:

```python
class RenderConfig:
    # Image output
    image_size = (512, 512)
    samples_per_handshape = 100

    # Camera variation
    camera_distance_range = (0.3, 0.5)
    camera_elevation_range = (-30, 60)
    camera_azimuth_range = (-45, 45)

    # Skin tones (R, G, B multipliers)
    skin_tones = [
        (1.0, 0.85, 0.75),   # Light
        (0.6, 0.45, 0.35),   # Medium-dark
        # Add more...
    ]

    # Render quality
    samples = 64  # Cycles samples
    use_denoising = True
```

### Add New Handshapes

Edit `asl_handshapes.py`:

```python
ASL_HANDSHAPES["NEW_SIGN"] = HandshapeConfig(
    name="NEW_SIGN",
    description="Description of the sign",
    pose=make_pose(
        index=[EXTENDED, EXTENDED, EXTENDED],
        middle=[FULLY_FLEXED, FULLY_FLEXED, FULLY_FLEXED],
        ring=[FULLY_FLEXED, FULLY_FLEXED, FULLY_FLEXED],
        pinky=[FULLY_FLEXED, FULLY_FLEXED, FULLY_FLEXED],
        thumb=THUMB_ALONGSIDE
    )
)
```

## Performance Tips

1. **Use GPU rendering**: Set `use_gpu = True` in RenderConfig
2. **Parallel rendering**: Use `--parallel 4` for multi-core systems
3. **Lower samples**: Use `--samples 50` for faster iteration
4. **Reduce quality**: Lower Cycles samples (32 instead of 64)

## Troubleshooting

### Blender not found

```bash
# Set path explicitly
export BLENDER_PATH=/path/to/blender
python batch_render.py --test
```

### GPU not detected

Check Blender preferences:
```python
# In mano_renderer.py, try different compute types:
prefs.compute_device_type = 'CUDA'   # NVIDIA
prefs.compute_device_type = 'OPTIX'  # NVIDIA RTX
prefs.compute_device_type = 'HIP'    # AMD
prefs.compute_device_type = 'METAL'  # Apple Silicon
```

### Bone mapping issues

If poses look wrong, check your model's bone names:
```bash
# List bones in your model
blender --background your_model.blend --python -c "
import bpy
arm = bpy.data.objects['Armature']
print([b.name for b in arm.pose.bones])
"
```

## Author

**Dawnena Key** / SonZo AI
Patent Pending 63/918,518
