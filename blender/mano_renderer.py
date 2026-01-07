"""
MANO Hand Model Renderer for Blender
=====================================
Generates synthetic ASL handshape training data using Blender.

Requirements:
- Blender 3.0+ with Python API
- MANO hand model (or compatible rigged hand mesh)

Usage:
    blender --background --python mano_renderer.py -- --handshape A --output ./output

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import bpy
import bmesh
import math
import json
import os
import sys
import random
from pathlib import Path
from mathutils import Vector, Euler, Matrix
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from asl_handshapes import ASL_HANDSHAPES, get_handshape, get_pose_array, FingerJoint
except ImportError:
    print("Warning: Could not import asl_handshapes. Using inline definitions.")
    ASL_HANDSHAPES = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class RenderConfig:
    """Configuration for rendering pipeline."""

    # Output settings
    output_dir: str = "./synthetic_data"
    image_format: str = "PNG"
    image_size: Tuple[int, int] = (512, 512)

    # Camera settings
    camera_distance_range: Tuple[float, float] = (0.3, 0.5)
    camera_elevation_range: Tuple[float, float] = (-30, 60)  # degrees
    camera_azimuth_range: Tuple[float, float] = (-45, 45)    # degrees

    # Lighting variations
    light_intensity_range: Tuple[float, float] = (800, 1500)
    light_color_temps: List[int] = [4500, 5500, 6500]  # Kelvin

    # Skin tone variations (R, G, B multipliers)
    skin_tones: List[Tuple[float, float, float]] = [
        (1.0, 0.85, 0.75),   # Light
        (0.9, 0.75, 0.65),   # Light-medium
        (0.75, 0.6, 0.5),    # Medium
        (0.6, 0.45, 0.35),   # Medium-dark
        (0.45, 0.3, 0.25),   # Dark
        (0.35, 0.22, 0.18),  # Very dark
    ]

    # Background options
    backgrounds: List[str] = ["solid", "gradient", "studio"]
    background_colors: List[Tuple[float, float, float]] = [
        (0.1, 0.1, 0.1),    # Dark gray
        (0.9, 0.9, 0.9),    # Light gray
        (0.2, 0.3, 0.4),    # Blue-gray
        (0.3, 0.25, 0.2),   # Warm brown
    ]

    # Samples per handshape
    samples_per_handshape: int = 100

    # Render settings
    use_gpu: bool = True
    samples: int = 64  # Cycles samples
    use_denoising: bool = True


# ==============================================================================
# MANO JOINT MAPPING
# ==============================================================================

# Map MANO joint indices to Blender bone names
# Adjust these names based on your actual rigged hand model
MANO_TO_BLENDER_BONES = {
    # Index finger
    0: "index_01",      # INDEX_MCP
    1: "index_02",      # INDEX_PIP
    2: "index_03",      # INDEX_DIP
    # Middle finger
    3: "middle_01",     # MIDDLE_MCP
    4: "middle_02",     # MIDDLE_PIP
    5: "middle_03",     # MIDDLE_DIP
    # Ring finger
    6: "ring_01",       # RING_MCP
    7: "ring_02",       # RING_PIP
    8: "ring_03",       # RING_DIP
    # Pinky
    9: "pinky_01",      # PINKY_MCP
    10: "pinky_02",     # PINKY_PIP
    11: "pinky_03",     # PINKY_DIP
    # Thumb
    12: "thumb_01",     # THUMB_CMC
    13: "thumb_02",     # THUMB_MCP
    14: "thumb_03",     # THUMB_IP
}

# Alternative naming conventions (try these if above doesn't work)
ALTERNATIVE_BONE_NAMES = {
    # Rigify style
    "rigify": {
        0: "f_index.01.L", 1: "f_index.02.L", 2: "f_index.03.L",
        3: "f_middle.01.L", 4: "f_middle.02.L", 5: "f_middle.03.L",
        6: "f_ring.01.L", 7: "f_ring.02.L", 8: "f_ring.03.L",
        9: "f_pinky.01.L", 10: "f_pinky.02.L", 11: "f_pinky.03.L",
        12: "thumb.01.L", 13: "thumb.02.L", 14: "thumb.03.L",
    },
    # MakeHuman style
    "makehuman": {
        0: "index1_L", 1: "index2_L", 2: "index3_L",
        3: "middle1_L", 4: "middle2_L", 5: "middle3_L",
        6: "ring1_L", 7: "ring2_L", 8: "ring3_L",
        9: "pinky1_L", 10: "pinky2_L", 11: "pinky3_L",
        12: "thumb1_L", 13: "thumb2_L", 14: "thumb3_L",
    },
    # MANO original
    "mano": {
        0: "index1", 1: "index2", 2: "index3",
        3: "middle1", 4: "middle2", 5: "middle3",
        6: "ring1", 7: "ring2", 8: "ring3",
        9: "pinky1", 10: "pinky2", 11: "pinky3",
        12: "thumb1", 13: "thumb2", 14: "thumb3",
    },
}


# ==============================================================================
# BLENDER SCENE SETUP
# ==============================================================================

class BlenderSceneSetup:
    """Set up Blender scene for rendering."""

    def __init__(self, config: RenderConfig):
        self.config = config
        self.hand_obj = None
        self.armature = None
        self.camera = None
        self.lights = []

    def clear_scene(self):
        """Remove all objects from scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Clear orphan data
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

    def setup_render_settings(self):
        """Configure render settings."""
        scene = bpy.context.scene

        # Use Cycles for realistic rendering
        scene.render.engine = 'CYCLES'

        # GPU rendering
        if self.config.use_gpu:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'  # or 'OPTIX', 'HIP', 'METAL'
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = 'GPU'

        # Quality settings
        scene.cycles.samples = self.config.samples
        scene.cycles.use_denoising = self.config.use_denoising

        # Output settings
        scene.render.resolution_x = self.config.image_size[0]
        scene.render.resolution_y = self.config.image_size[1]
        scene.render.image_settings.file_format = self.config.image_format
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.film_transparent = True

    def create_simple_hand(self) -> bpy.types.Object:
        """
        Create a simple rigged hand mesh for testing.
        In production, replace this with MANO model import.
        """
        # Create armature
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.context.active_object
        armature.name = "HandArmature"

        # Get armature data
        arm_data = armature.data
        arm_data.name = "HandArmatureData"

        # Remove default bone
        for bone in arm_data.edit_bones:
            arm_data.edit_bones.remove(bone)

        # Create hand bones
        # Palm/Wrist
        wrist = arm_data.edit_bones.new("wrist")
        wrist.head = Vector((0, 0, 0))
        wrist.tail = Vector((0, 0.05, 0))

        # Create finger bones
        finger_names = ["index", "middle", "ring", "pinky", "thumb"]
        finger_offsets = [
            Vector((0.02, 0.08, 0)),    # Index
            Vector((0.007, 0.085, 0)),  # Middle
            Vector((-0.007, 0.08, 0)),  # Ring
            Vector((-0.02, 0.07, 0)),   # Pinky
            Vector((0.03, 0.03, 0)),    # Thumb
        ]
        bone_lengths = [0.03, 0.02, 0.015]  # Per segment

        for finger_idx, (finger_name, offset) in enumerate(zip(finger_names, finger_offsets)):
            parent = wrist
            head_pos = wrist.tail + offset

            for seg_idx in range(3):
                bone_name = f"{finger_name}_{seg_idx+1:02d}"
                bone = arm_data.edit_bones.new(bone_name)

                # Position bone
                bone.head = head_pos

                # Direction (roughly upward, thumb is different)
                if finger_name == "thumb":
                    direction = Vector((0.5, 0.5, 0.3)).normalized()
                else:
                    direction = Vector((0, 1, 0.1)).normalized()

                bone.tail = head_pos + direction * bone_lengths[seg_idx]
                bone.parent = parent

                parent = bone
                head_pos = bone.tail

        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create mesh
        bpy.ops.mesh.primitive_cube_add(size=0.1)
        hand_mesh = bpy.context.active_object
        hand_mesh.name = "HandMesh"
        hand_mesh.scale = (0.08, 0.15, 0.03)
        bpy.ops.object.transform_apply(scale=True)

        # Parent mesh to armature
        hand_mesh.parent = armature
        modifier = hand_mesh.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = armature

        # Create skin material
        self.create_skin_material(hand_mesh)

        self.armature = armature
        self.hand_obj = hand_mesh

        return hand_mesh

    def load_mano_model(self, model_path: str) -> bpy.types.Object:
        """
        Load MANO hand model from file.
        Supports: .fbx, .obj, .blend
        """
        ext = Path(model_path).suffix.lower()

        if ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=model_path)
        elif ext == '.obj':
            bpy.ops.import_scene.obj(filepath=model_path)
        elif ext == '.blend':
            # Append from .blend file
            with bpy.data.libraries.load(model_path) as (data_from, data_to):
                data_to.objects = data_from.objects
            for obj in data_to.objects:
                bpy.context.collection.objects.link(obj)
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        # Find armature and mesh
        for obj in bpy.context.selected_objects:
            if obj.type == 'ARMATURE':
                self.armature = obj
            elif obj.type == 'MESH':
                self.hand_obj = obj

        if not self.armature:
            raise ValueError("No armature found in model")
        if not self.hand_obj:
            raise ValueError("No mesh found in model")

        return self.hand_obj

    def create_skin_material(self, obj: bpy.types.Object, skin_tone_idx: int = 0):
        """Create realistic skin material."""
        mat = bpy.data.materials.new(name="SkinMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (400, 0)

        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)

        # Set skin tone
        skin_tone = self.config.skin_tones[skin_tone_idx % len(self.config.skin_tones)]
        principled.inputs['Base Color'].default_value = (*skin_tone, 1.0)

        # Subsurface scattering for realistic skin
        principled.inputs['Subsurface Weight'].default_value = 0.3
        principled.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)

        # Roughness
        principled.inputs['Roughness'].default_value = 0.5

        # Connect
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        # Assign material
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def setup_camera(self) -> bpy.types.Object:
        """Create and position camera."""
        bpy.ops.object.camera_add()
        self.camera = bpy.context.active_object
        self.camera.name = "RenderCamera"

        bpy.context.scene.camera = self.camera

        return self.camera

    def randomize_camera(self):
        """Randomize camera position within configured ranges."""
        if not self.camera:
            return

        # Random distance
        distance = random.uniform(*self.config.camera_distance_range)

        # Random angles
        elevation = math.radians(random.uniform(*self.config.camera_elevation_range))
        azimuth = math.radians(random.uniform(*self.config.camera_azimuth_range))

        # Calculate position
        x = distance * math.cos(elevation) * math.sin(azimuth)
        y = -distance * math.cos(elevation) * math.cos(azimuth)
        z = distance * math.sin(elevation) + 0.1  # Offset for hand height

        self.camera.location = Vector((x, y, z))

        # Point at hand
        direction = Vector((0, 0, 0.1)) - self.camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()

    def setup_lighting(self):
        """Create studio-style lighting."""
        # Key light
        bpy.ops.object.light_add(type='AREA', location=(0.5, -0.5, 0.5))
        key_light = bpy.context.active_object
        key_light.name = "KeyLight"
        key_light.data.energy = 1000
        key_light.data.size = 0.5
        self.lights.append(key_light)

        # Fill light
        bpy.ops.object.light_add(type='AREA', location=(-0.4, -0.3, 0.3))
        fill_light = bpy.context.active_object
        fill_light.name = "FillLight"
        fill_light.data.energy = 500
        fill_light.data.size = 0.3
        self.lights.append(fill_light)

        # Rim light
        bpy.ops.object.light_add(type='AREA', location=(0, 0.5, 0.5))
        rim_light = bpy.context.active_object
        rim_light.name = "RimLight"
        rim_light.data.energy = 300
        rim_light.data.size = 0.2
        self.lights.append(rim_light)

    def randomize_lighting(self):
        """Randomize lighting conditions."""
        for light in self.lights:
            # Random intensity variation
            base_energy = light.data.energy
            light.data.energy = base_energy * random.uniform(0.7, 1.3)

            # Random color temperature
            temp = random.choice(self.config.light_color_temps)
            # Convert Kelvin to RGB (simplified)
            if temp < 5500:
                light.data.color = (1.0, 0.95, 0.9)  # Warm
            elif temp > 5500:
                light.data.color = (0.9, 0.95, 1.0)  # Cool
            else:
                light.data.color = (1.0, 1.0, 1.0)   # Neutral

    def setup_background(self, bg_type: str = "solid", color_idx: int = 0):
        """Set up scene background."""
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world

        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        nodes.clear()

        output = nodes.new('ShaderNodeOutputWorld')
        output.location = (400, 0)

        if bg_type == "solid":
            bg = nodes.new('ShaderNodeBackground')
            bg.location = (0, 0)
            color = self.config.background_colors[color_idx % len(self.config.background_colors)]
            bg.inputs['Color'].default_value = (*color, 1.0)
            bg.inputs['Strength'].default_value = 1.0
            links.new(bg.outputs['Background'], output.inputs['Surface'])

        elif bg_type == "gradient":
            # Gradient background
            bg = nodes.new('ShaderNodeBackground')
            bg.location = (0, 0)

            gradient = nodes.new('ShaderNodeTexGradient')
            gradient.location = (-400, 0)

            mapping = nodes.new('ShaderNodeMapping')
            mapping.location = (-600, 0)

            texcoord = nodes.new('ShaderNodeTexCoord')
            texcoord.location = (-800, 0)

            colorramp = nodes.new('ShaderNodeValToRGB')
            colorramp.location = (-200, 0)

            links.new(texcoord.outputs['Window'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
            links.new(gradient.outputs['Fac'], colorramp.inputs['Fac'])
            links.new(colorramp.outputs['Color'], bg.inputs['Color'])
            links.new(bg.outputs['Background'], output.inputs['Surface'])


# ==============================================================================
# POSE APPLICATION
# ==============================================================================

class PoseApplicator:
    """Apply MANO pose parameters to Blender armature."""

    def __init__(self, armature: bpy.types.Object, bone_naming: str = "default"):
        self.armature = armature
        self.bone_map = self._get_bone_map(bone_naming)

    def _get_bone_map(self, naming: str) -> Dict[int, str]:
        """Get bone name mapping based on naming convention."""
        if naming in ALTERNATIVE_BONE_NAMES:
            return ALTERNATIVE_BONE_NAMES[naming]
        return MANO_TO_BLENDER_BONES

    def _detect_bone_naming(self) -> str:
        """Auto-detect bone naming convention."""
        if not self.armature:
            return "default"

        bone_names = [b.name for b in self.armature.pose.bones]

        # Check for each naming convention
        for name, bone_map in ALTERNATIVE_BONE_NAMES.items():
            if any(b in bone_names for b in bone_map.values()):
                print(f"Detected bone naming: {name}")
                return name

        return "default"

    def apply_pose(self, pose: List[List[float]], mirror: bool = False):
        """
        Apply MANO pose to armature.

        Args:
            pose: 15x3 array of joint rotations [flexion, abduction, rotation]
            mirror: If True, apply to right hand (mirror rotations)
        """
        if not self.armature:
            raise ValueError("No armature set")

        # Enter pose mode
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')

        for joint_idx, rotations in enumerate(pose):
            bone_name = self.bone_map.get(joint_idx)
            if not bone_name:
                continue

            # Try to find bone with different naming
            bone = self.armature.pose.bones.get(bone_name)
            if not bone:
                # Try alternative names
                for naming, bone_map in ALTERNATIVE_BONE_NAMES.items():
                    alt_name = bone_map.get(joint_idx)
                    if alt_name:
                        bone = self.armature.pose.bones.get(alt_name)
                        if bone:
                            break

            if not bone:
                print(f"Warning: Bone not found for joint {joint_idx}: {bone_name}")
                continue

            # Apply rotation
            flexion, abduction, rotation = rotations

            # Mirror for right hand
            if mirror:
                abduction = -abduction
                rotation = -rotation

            # Set rotation (Euler XYZ)
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = Euler((flexion, abduction, rotation), 'XYZ')

        # Update scene
        bpy.context.view_layer.update()

        # Return to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

    def apply_handshape(self, handshape_name: str, mirror: bool = False):
        """Apply a named handshape from ASL_HANDSHAPES."""
        if not ASL_HANDSHAPES:
            raise ValueError("ASL handshapes not loaded")

        config = ASL_HANDSHAPES.get(handshape_name.upper())
        if not config:
            raise ValueError(f"Unknown handshape: {handshape_name}")

        self.apply_pose(config.pose, mirror=mirror)

        return config

    def reset_pose(self):
        """Reset armature to rest pose."""
        if not self.armature:
            return

        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')


# ==============================================================================
# RENDERING PIPELINE
# ==============================================================================

class RenderingPipeline:
    """Main rendering pipeline for synthetic data generation."""

    def __init__(self, config: RenderConfig):
        self.config = config
        self.scene_setup = BlenderSceneSetup(config)
        self.pose_applicator = None

        # Metadata for all renders
        self.metadata = {
            "version": "1.0",
            "config": {
                "image_size": config.image_size,
                "samples_per_handshape": config.samples_per_handshape,
            },
            "samples": []
        }

    def setup(self, model_path: Optional[str] = None):
        """Initialize the rendering pipeline."""
        # Clear scene
        self.scene_setup.clear_scene()

        # Set up render settings
        self.scene_setup.setup_render_settings()

        # Load or create hand model
        if model_path and Path(model_path).exists():
            self.scene_setup.load_mano_model(model_path)
        else:
            print("Creating simple hand model for testing...")
            self.scene_setup.create_simple_hand()

        # Set up camera and lighting
        self.scene_setup.setup_camera()
        self.scene_setup.setup_lighting()
        self.scene_setup.setup_background()

        # Initialize pose applicator
        self.pose_applicator = PoseApplicator(self.scene_setup.armature)

    def render_handshape(
        self,
        handshape_name: str,
        output_dir: Path,
        num_samples: int = None
    ) -> List[Dict]:
        """
        Render multiple variations of a handshape.

        Args:
            handshape_name: Name of the handshape (e.g., "A", "B", "ILY")
            output_dir: Directory to save renders
            num_samples: Number of variations to render

        Returns:
            List of sample metadata dictionaries
        """
        if num_samples is None:
            num_samples = self.config.samples_per_handshape

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = []

        for i in range(num_samples):
            sample_id = f"{handshape_name}_{i:04d}"

            # Apply handshape pose
            try:
                handshape_config = self.pose_applicator.apply_handshape(handshape_name)
            except Exception as e:
                print(f"Error applying pose: {e}")
                continue

            # Randomize variations
            skin_tone_idx = random.randint(0, len(self.config.skin_tones) - 1)
            self.scene_setup.create_skin_material(
                self.scene_setup.hand_obj,
                skin_tone_idx
            )

            self.scene_setup.randomize_camera()
            self.scene_setup.randomize_lighting()

            # Random background
            bg_type = random.choice(self.config.backgrounds)
            bg_color_idx = random.randint(0, len(self.config.background_colors) - 1)
            self.scene_setup.setup_background(bg_type, bg_color_idx)

            # Set output path
            output_path = output_dir / f"{sample_id}.png"
            bpy.context.scene.render.filepath = str(output_path)

            # Render
            bpy.ops.render.render(write_still=True)

            # Get keypoints (if available)
            keypoints_2d = self._get_2d_keypoints()
            keypoints_3d = self._get_3d_keypoints()

            # Record metadata
            sample_meta = {
                "id": len(self.metadata["samples"]),
                "sample_id": sample_id,
                "handshape": handshape_name,
                "filepath": str(output_path.relative_to(output_dir.parent)),
                "skin_tone_idx": skin_tone_idx,
                "background_type": bg_type,
                "keypoints_2d": keypoints_2d,
                "keypoints_3d": keypoints_3d,
            }

            samples.append(sample_meta)
            self.metadata["samples"].append(sample_meta)

            print(f"Rendered {sample_id} ({i+1}/{num_samples})")

        return samples

    def render_all_handshapes(self, output_dir: str):
        """Render all defined handshapes."""
        if not ASL_HANDSHAPES:
            raise ValueError("ASL handshapes not loaded")

        output_dir = Path(output_dir)

        for handshape_name in ASL_HANDSHAPES.keys():
            print(f"\n=== Rendering handshape: {handshape_name} ===")
            handshape_dir = output_dir / handshape_name
            self.render_handshape(handshape_name, handshape_dir)

        # Save metadata
        self.save_metadata(output_dir / "metadata.json")

        # Save label map
        label_map = {name: idx for idx, name in enumerate(sorted(ASL_HANDSHAPES.keys()))}
        with open(output_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)

    def _get_2d_keypoints(self) -> Optional[List[List[float]]]:
        """Project 3D bone positions to 2D screen coordinates."""
        if not self.scene_setup.armature or not self.scene_setup.camera:
            return None

        keypoints = []
        scene = bpy.context.scene

        for joint_idx in range(15):
            bone_name = MANO_TO_BLENDER_BONES.get(joint_idx)
            if not bone_name:
                keypoints.append([0, 0])
                continue

            bone = self.scene_setup.armature.pose.bones.get(bone_name)
            if not bone:
                keypoints.append([0, 0])
                continue

            # Get world position
            world_pos = self.scene_setup.armature.matrix_world @ bone.head

            # Project to camera
            co_2d = bpy_extras.object_utils.world_to_camera_view(
                scene, self.scene_setup.camera, world_pos
            )

            # Convert to pixel coordinates
            render = scene.render
            x = co_2d.x * render.resolution_x
            y = (1 - co_2d.y) * render.resolution_y  # Flip Y

            keypoints.append([x, y])

        return keypoints

    def _get_3d_keypoints(self) -> Optional[List[List[float]]]:
        """Get 3D bone positions."""
        if not self.scene_setup.armature:
            return None

        keypoints = []

        for joint_idx in range(15):
            bone_name = MANO_TO_BLENDER_BONES.get(joint_idx)
            if not bone_name:
                keypoints.append([0, 0, 0])
                continue

            bone = self.scene_setup.armature.pose.bones.get(bone_name)
            if not bone:
                keypoints.append([0, 0, 0])
                continue

            # Get world position
            world_pos = self.scene_setup.armature.matrix_world @ bone.head
            keypoints.append([world_pos.x, world_pos.y, world_pos.z])

        return keypoints

    def save_metadata(self, filepath: str):
        """Save all metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata to {filepath}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point for command-line usage."""
    import argparse

    # Parse arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Render synthetic ASL handshapes')
    parser.add_argument('--handshape', type=str, help='Single handshape to render')
    parser.add_argument('--all', action='store_true', help='Render all handshapes')
    parser.add_argument('--output', type=str, default='./synthetic_data', help='Output directory')
    parser.add_argument('--model', type=str, help='Path to hand model file')
    parser.add_argument('--samples', type=int, default=10, help='Samples per handshape')
    parser.add_argument('--size', type=int, default=512, help='Image size')
    parser.add_argument('--test', action='store_true', help='Run test render')

    args = parser.parse_args(argv)

    # Configure
    config = RenderConfig()
    config.output_dir = args.output
    config.samples_per_handshape = args.samples
    config.image_size = (args.size, args.size)

    # Initialize pipeline
    pipeline = RenderingPipeline(config)
    pipeline.setup(args.model)

    if args.test:
        # Quick test render
        print("Running test render...")
        pipeline.render_handshape("A", Path(args.output) / "test", num_samples=3)
        print("Test complete!")

    elif args.all:
        # Render all handshapes
        pipeline.render_all_handshapes(args.output)

    elif args.handshape:
        # Render specific handshape
        pipeline.render_handshape(
            args.handshape.upper(),
            Path(args.output) / args.handshape.upper()
        )

    else:
        print("No action specified. Use --handshape, --all, or --test")
        parser.print_help()


# Allow running as Blender script
if __name__ == "__main__":
    try:
        import bpy_extras.object_utils
    except ImportError:
        print("This script must be run from within Blender")
        print("Usage: blender --background --python mano_renderer.py -- --test")
        sys.exit(1)

    main()
