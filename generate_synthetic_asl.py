#!/usr/bin/env python3
"""
Synthetic ASL Data Generator for Blender
=========================================
Generates labeled synthetic ASL hand images for training sign language recognition models.

Run with: blender --background --python generate_synthetic_asl.py -- [options]

Author: Dawnena Key / SonZo AI
License: Proprietary - Patent Pending
"""

import bpy
import bmesh
import numpy as np
import json
import os
import sys
import base64
import argparse
import random
import math
from pathlib import Path
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from mathutils import Vector, Euler, Matrix

# Add project path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from asl_handshapes import (
    ASL_HANDSHAPES, 
    get_handshape, 
    get_pose_array,
    get_all_handshapes,
    HandshapeConfig
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Generation configuration with defaults."""
    
    def __init__(self):
        self.output_dir: str = "/tmp/synthetic_asl_data"
        self.image_size: Tuple[int, int] = (512, 512)
        self.samples_per_handshape: int = 100
        self.export_format: str = "PNG"  # "PNG" or "BASE64"
        self.render_engine: str = "CYCLES"  # "CYCLES" or "EEVEE"
        self.render_samples: int = 128
        self.use_gpu: bool = True
        
        # Target handshapes (None = all)
        self.target_handshapes: Optional[List[str]] = None
        
        # Skin tone RGB values (realistic range)
        self.skin_tones: List[Tuple[float, float, float]] = [
            (0.95, 0.82, 0.72),   # Very light
            (0.90, 0.75, 0.62),   # Light
            (0.82, 0.64, 0.50),   # Medium light
            (0.70, 0.50, 0.38),   # Medium
            (0.58, 0.40, 0.30),   # Medium dark
            (0.45, 0.30, 0.22),   # Dark
            (0.35, 0.22, 0.16),   # Very dark
        ]
        
        # Camera variation ranges
        self.camera_distance_range: Tuple[float, float] = (0.35, 0.55)
        self.camera_elevation_range: Tuple[float, float] = (-30, 45)  # degrees
        self.camera_azimuth_range: Tuple[float, float] = (-60, 60)    # degrees
        
        # Lighting variation
        self.lighting_intensity_range: Tuple[float, float] = (0.6, 1.8)
        self.lighting_color_temp_range: Tuple[int, int] = (4000, 7000)  # Kelvin
        
        # Background options
        self.background_types: List[str] = ["solid", "gradient", "noise"]
        self.background_colors: List[Tuple[float, float, float]] = [
            (1.0, 1.0, 1.0),     # White
            (0.95, 0.95, 0.95),  # Off-white
            (0.85, 0.85, 0.85),  # Light gray
            (0.7, 0.7, 0.7),     # Medium gray
            (0.3, 0.3, 0.3),     # Dark gray
            (0.1, 0.1, 0.1),     # Near black
            (0.95, 0.90, 0.85),  # Warm cream
            (0.85, 0.90, 0.95),  # Cool blue-gray
        ]
        
        # Hand pose variation (add noise to base pose)
        self.pose_noise_std: float = 0.05  # radians
        
        # Wrist rotation ranges
        self.wrist_rotation_range: Tuple[float, float] = (-0.3, 0.3)  # radians
        
        # Export keypoints
        self.export_keypoints: bool = True
        
        # Random seed for reproducibility
        self.seed: Optional[int] = None


CONFIG = Config()


# ============================================================================
# SCENE SETUP
# ============================================================================

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_settings():
    """Configure render settings for quality synthetic data."""
    scene = bpy.context.scene
    
    # Render engine
    if CONFIG.render_engine == "CYCLES":
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU' if CONFIG.use_gpu else 'CPU'
        scene.cycles.samples = CONFIG.render_samples
        scene.cycles.use_denoising = True
        
        # Try to enable GPU
        if CONFIG.use_gpu:
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA'
                prefs.get_devices()
                for device in prefs.devices:
                    device.use = True
            except Exception as e:
                print(f"GPU setup warning: {e}")
    else:
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = CONFIG.render_samples
    
    # Resolution
    scene.render.resolution_x = CONFIG.image_size[0]
    scene.render.resolution_y = CONFIG.image_size[1]
    scene.render.resolution_percentage = 100
    
    # Output format
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    
    # Film settings
    scene.render.film_transparent = False


def create_camera() -> bpy.types.Object:
    """Create and configure the main camera."""
    bpy.ops.object.camera_add(location=(0, -0.5, 0))
    camera = bpy.context.object
    camera.name = "MainCamera"
    
    # Set camera properties
    camera.data.lens = 50  # mm
    camera.data.sensor_width = 36  # Full frame
    camera.data.clip_start = 0.01
    camera.data.clip_end = 10
    
    # Point at origin using track constraint
    track = camera.constraints.new(type='TRACK_TO')
    track.target = None  # Will track origin
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera


def create_lighting() -> Dict[str, bpy.types.Object]:
    """Create a three-point lighting setup."""
    lights = {}
    
    # Key light (main light)
    bpy.ops.object.light_add(type='AREA', location=(0.8, -0.8, 1.2))
    key_light = bpy.context.object
    key_light.name = "KeyLight"
    key_light.data.energy = 150
    key_light.data.size = 1.5
    key_light.data.color = (1.0, 0.98, 0.95)  # Slightly warm
    lights['key'] = key_light
    
    # Fill light (softer, opposite side)
    bpy.ops.object.light_add(type='AREA', location=(-1.0, -0.6, 0.8))
    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 80
    fill_light.data.size = 2.0
    fill_light.data.color = (0.95, 0.98, 1.0)  # Slightly cool
    lights['fill'] = fill_light
    
    # Rim/back light (separation from background)
    bpy.ops.object.light_add(type='AREA', location=(0.3, 0.8, 1.0))
    rim_light = bpy.context.object
    rim_light.name = "RimLight"
    rim_light.data.energy = 60
    rim_light.data.size = 1.0
    lights['rim'] = rim_light
    
    # Ambient/bounce light from below
    bpy.ops.object.light_add(type='AREA', location=(0, -0.3, -0.5))
    bounce_light = bpy.context.object
    bounce_light.name = "BounceLight"
    bounce_light.data.energy = 30
    bounce_light.data.size = 2.5
    lights['bounce'] = bounce_light
    
    return lights


# ============================================================================
# HAND MODEL CREATION
# ============================================================================

def create_simple_hand_mesh() -> bpy.types.Object:
    """
    Create a simple but recognizable hand mesh with armature.
    This is a placeholder - replace with MANO model loading in production.
    """
    # Create a basic hand shape using primitives
    bpy.ops.mesh.primitive_cube_add(size=0.08, location=(0, 0, 0))
    palm = bpy.context.object
    palm.name = "Hand_Palm"
    palm.scale = (1.0, 0.6, 0.15)
    bpy.ops.object.transform_apply(scale=True)
    
    # Add fingers as cylinders
    finger_data = [
        # (name, base_pos, length, radius)
        ("Thumb", (-0.035, 0.01, 0.01), 0.04, 0.008),
        ("Index", (-0.02, 0.04, 0.005), 0.055, 0.007),
        ("Middle", (0.0, 0.045, 0.005), 0.06, 0.007),
        ("Ring", (0.018, 0.042, 0.005), 0.055, 0.007),
        ("Pinky", (0.035, 0.035, 0.005), 0.045, 0.006),
    ]
    
    finger_objects = []
    for name, pos, length, radius in finger_data:
        # Create finger with 3 segments
        for seg in range(3):
            seg_length = length / 3 * (1 - seg * 0.1)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=radius * (1 - seg * 0.15),
                depth=seg_length,
                location=(0, 0, 0)
            )
            segment = bpy.context.object
            segment.name = f"Hand_{name}_{seg}"
            finger_objects.append(segment)
    
    # Join all parts
    bpy.ops.object.select_all(action='DESELECT')
    palm.select_set(True)
    for obj in finger_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = palm
    bpy.ops.object.join()
    
    hand = bpy.context.object
    hand.name = "Hand"
    
    return hand


def create_hand_armature() -> Tuple[bpy.types.Object, bpy.types.Object]:
    """
    Create an armature for hand posing.
    Returns (hand_mesh, armature).
    """
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "HandArmature"
    
    arm = armature.data
    arm.name = "HandArmatureData"
    
    # Clear default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Bone definitions: (name, head, tail, parent)
    bones_def = [
        # Wrist (root)
        ("Wrist", (0, 0, 0), (0, 0.02, 0), None),
        
        # Index finger
        ("Index_MCP", (-0.02, 0.03, 0), (-0.02, 0.05, 0), "Wrist"),
        ("Index_PIP", (-0.02, 0.05, 0), (-0.02, 0.065, 0), "Index_MCP"),
        ("Index_DIP", (-0.02, 0.065, 0), (-0.02, 0.08, 0), "Index_PIP"),
        
        # Middle finger
        ("Middle_MCP", (0, 0.035, 0), (0, 0.055, 0), "Wrist"),
        ("Middle_PIP", (0, 0.055, 0), (0, 0.072, 0), "Middle_MCP"),
        ("Middle_DIP", (0, 0.072, 0), (0, 0.088, 0), "Middle_PIP"),
        
        # Ring finger
        ("Ring_MCP", (0.018, 0.032, 0), (0.018, 0.050, 0), "Wrist"),
        ("Ring_PIP", (0.018, 0.050, 0), (0.018, 0.065, 0), "Ring_MCP"),
        ("Ring_DIP", (0.018, 0.065, 0), (0.018, 0.078, 0), "Ring_PIP"),
        
        # Pinky
        ("Pinky_MCP", (0.035, 0.028, 0), (0.035, 0.042, 0), "Wrist"),
        ("Pinky_PIP", (0.035, 0.042, 0), (0.035, 0.054, 0), "Pinky_MCP"),
        ("Pinky_DIP", (0.035, 0.054, 0), (0.035, 0.064, 0), "Pinky_PIP"),
        
        # Thumb
        ("Thumb_CMC", (-0.035, 0.005, 0.005), (-0.045, 0.015, 0.01), "Wrist"),
        ("Thumb_MCP", (-0.045, 0.015, 0.01), (-0.052, 0.028, 0.012), "Thumb_CMC"),
        ("Thumb_IP", (-0.052, 0.028, 0.012), (-0.058, 0.04, 0.014), "Thumb_MCP"),
    ]
    
    # Create bones
    created_bones = {}
    for name, head, tail, parent_name in bones_def:
        bone = arm.edit_bones.new(name)
        bone.head = Vector(head)
        bone.tail = Vector(tail)
        if parent_name and parent_name in created_bones:
            bone.parent = created_bones[parent_name]
            bone.use_connect = (parent_name != "Wrist")
        created_bones[name] = bone
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create hand mesh
    hand = create_simplified_hand_geometry()
    
    # Parent mesh to armature with automatic weights
    hand.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    return hand, armature


def create_simplified_hand_geometry() -> bpy.types.Object:
    """Create a more anatomically-shaped hand mesh."""
    # Create base mesh
    mesh = bpy.data.meshes.new("HandMesh")
    obj = bpy.data.objects.new("Hand", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    bm = bmesh.new()
    
    # Palm - create as a scaled cube
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.07, 0.05, 0.015), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0.015, 0), verts=bm.verts)
    
    # Add fingers as extruded cylinders
    finger_specs = [
        # (x_offset, y_start, lengths, radius)
        (-0.02, 0.04, [0.02, 0.015, 0.012], 0.007),   # Index
        (0.0, 0.045, [0.022, 0.017, 0.013], 0.007),   # Middle  
        (0.018, 0.042, [0.020, 0.015, 0.012], 0.007), # Ring
        (0.035, 0.035, [0.016, 0.012, 0.010], 0.006), # Pinky
    ]
    
    for x_off, y_start, lengths, radius in finger_specs:
        y_pos = y_start
        for i, length in enumerate(lengths):
            # Create cylinder for finger segment
            ret = bmesh.ops.create_cone(
                bm,
                segments=8,
                radius1=radius * (1 - i * 0.1),
                radius2=radius * (1 - (i + 1) * 0.1) * 0.9,
                depth=length
            )
            # Move to position
            for v in ret['verts']:
                v.co.x += x_off
                v.co.y += y_pos + length / 2
            y_pos += length
    
    # Thumb (angled)
    thumb_specs = [
        ([-0.04, 0.01, 0.008], 0.012, 0.008),
        ([-0.05, 0.022, 0.012], 0.010, 0.007),
        ([-0.058, 0.035, 0.014], 0.008, 0.006),
    ]
    
    for pos, length, radius in thumb_specs:
        ret = bmesh.ops.create_cone(bm, segments=8, radius1=radius, radius2=radius * 0.85, depth=length)
        for v in ret['verts']:
            v.co.x += pos[0]
            v.co.y += pos[1]
            v.co.z += pos[2]
    
    bm.to_mesh(mesh)
    bm.free()
    
    # Smooth shading
    for poly in mesh.polygons:
        poly.use_smooth = True
    
    # Add subdivision modifier for smoother appearance
    mod = obj.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels = 1
    mod.render_levels = 2
    
    return obj


# ============================================================================
# MATERIALS
# ============================================================================

def create_skin_material(color: Tuple[float, float, float]) -> bpy.types.Material:
    """Create a realistic skin material with subsurface scattering."""
    mat = bpy.data.materials.new(name="SkinMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create shader nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    
    # Skin shader settings
    principled.inputs['Base Color'].default_value = (*color, 1.0)
    principled.inputs['Roughness'].default_value = 0.45
    principled.inputs['Specular IOR Level'].default_value = 0.5
    
    # Subsurface scattering for realistic skin
    principled.inputs['Subsurface Weight'].default_value = 0.3
    principled.inputs['Subsurface Radius'].default_value = (0.8, 0.3, 0.1)
    principled.inputs['Subsurface Scale'].default_value = 0.01
    
    # Connect nodes
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def apply_material(obj: bpy.types.Object, material: bpy.types.Material):
    """Apply material to object."""
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


# ============================================================================
# VARIATIONS / RANDOMIZATION
# ============================================================================

def randomize_camera(camera: bpy.types.Object, seed: int):
    """Randomize camera position within configured ranges."""
    rng = np.random.RandomState(seed)
    
    # Random distance
    distance = rng.uniform(*CONFIG.camera_distance_range)
    
    # Random elevation and azimuth angles
    elevation = np.radians(rng.uniform(*CONFIG.camera_elevation_range))
    azimuth = np.radians(rng.uniform(*CONFIG.camera_azimuth_range))
    
    # Convert spherical to Cartesian coordinates
    x = distance * np.cos(elevation) * np.sin(azimuth)
    y = -distance * np.cos(elevation) * np.cos(azimuth)
    z = distance * np.sin(elevation)
    
    camera.location = (x, y, z)
    
    # Point camera at origin (hand center)
    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def randomize_lighting(lights: Dict[str, bpy.types.Object], seed: int):
    """Randomize lighting intensity and color temperature."""
    rng = np.random.RandomState(seed)
    
    # Overall intensity multiplier
    intensity_mult = rng.uniform(*CONFIG.lighting_intensity_range)
    
    # Color temperature variation
    color_temp = rng.uniform(*CONFIG.lighting_color_temp_range)
    
    # Convert color temp to RGB (simplified approximation)
    if color_temp < 5500:
        # Warmer (more red/yellow)
        r = 1.0
        g = 0.9 + (color_temp - 4000) / 15000
        b = 0.8 + (color_temp - 4000) / 10000
    else:
        # Cooler (more blue)
        r = 1.0 - (color_temp - 5500) / 15000
        g = 0.98 - (color_temp - 5500) / 30000
        b = 1.0
    
    # Apply to lights with individual variation
    base_energies = {'key': 150, 'fill': 80, 'rim': 60, 'bounce': 30}
    
    for name, light in lights.items():
        if name in base_energies:
            # Individual random variation
            individual_mult = rng.uniform(0.8, 1.2)
            light.data.energy = base_energies[name] * intensity_mult * individual_mult
            
            # Slight color variation per light
            light.data.color = (
                r * rng.uniform(0.95, 1.05),
                g * rng.uniform(0.95, 1.05),
                b * rng.uniform(0.95, 1.05)
            )


def set_background(bg_type: str, seed: int):
    """Set the scene background."""
    rng = np.random.RandomState(seed)
    world = bpy.context.scene.world
    
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create output node
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)
    
    # Choose background color
    bg_color = CONFIG.background_colors[rng.randint(len(CONFIG.background_colors))]
    
    if bg_type == "solid":
        # Simple solid color background
        background = nodes.new('ShaderNodeBackground')
        background.location = (0, 0)
        background.inputs['Color'].default_value = (*bg_color, 1.0)
        background.inputs['Strength'].default_value = 1.0
        links.new(background.outputs['Background'], output.inputs['Surface'])
        
    elif bg_type == "gradient":
        # Gradient background
        background = nodes.new('ShaderNodeBackground')
        background.location = (200, 0)
        
        gradient = nodes.new('ShaderNodeTexGradient')
        gradient.location = (-200, 0)
        gradient.gradient_type = 'LINEAR'
        
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-400, 0)
        
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (0, 0)
        
        # Set gradient colors
        ramp.color_ramp.elements[0].color = (*bg_color, 1.0)
        lighter = tuple(min(1.0, c + 0.2) for c in bg_color)
        ramp.color_ramp.elements[1].color = (*lighter, 1.0)
        
        links.new(coord.outputs['Window'], gradient.inputs['Vector'])
        links.new(gradient.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], output.inputs['Surface'])
        
    elif bg_type == "noise":
        # Noisy/textured background
        background = nodes.new('ShaderNodeBackground')
        background.location = (200, 0)
        
        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (-200, 0)
        noise.inputs['Scale'].default_value = rng.uniform(10, 50)
        noise.inputs['Detail'].default_value = 2.0
        
        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (0, 0)
        mix.inputs['Fac'].default_value = 0.1  # Subtle noise
        mix.inputs['Color1'].default_value = (*bg_color, 1.0)
        
        links.new(noise.outputs['Fac'], mix.inputs['Color2'])
        links.new(mix.outputs['Color'], background.inputs['Color'])
        links.new(background.outputs['Background'], output.inputs['Surface'])


def apply_handshape_pose(armature: bpy.types.Object, handshape_name: str, seed: int):
    """Apply a handshape pose to the hand armature with optional noise."""
    config = get_handshape(handshape_name)
    pose_array = get_pose_array(handshape_name)
    
    rng = np.random.RandomState(seed)
    
    # Add small random noise to pose for variation
    if CONFIG.pose_noise_std > 0:
        noise = rng.normal(0, CONFIG.pose_noise_std, pose_array.shape)
        pose_array = pose_array + noise.astype(np.float32)
    
    # Map pose indices to bone names
    bone_mapping = [
        "Index_MCP", "Index_PIP", "Index_DIP",
        "Middle_MCP", "Middle_PIP", "Middle_DIP",
        "Ring_MCP", "Ring_PIP", "Ring_DIP",
        "Pinky_MCP", "Pinky_PIP", "Pinky_DIP",
        "Thumb_CMC", "Thumb_MCP", "Thumb_IP",
    ]
    
    # Switch to pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Apply rotations to bones
    for i, bone_name in enumerate(bone_mapping):
        if bone_name in armature.pose.bones:
            bone = armature.pose.bones[bone_name]
            rotation = pose_array[i]
            
            # Apply as Euler rotation (XYZ)
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = Euler((rotation[0], rotation[1], rotation[2]), 'XYZ')
    
    # Add wrist rotation variation
    if "Wrist" in armature.pose.bones:
        wrist = armature.pose.bones["Wrist"]
        wrist.rotation_mode = 'XYZ'
        wrist.rotation_euler = Euler((
            rng.uniform(*CONFIG.wrist_rotation_range),
            rng.uniform(*CONFIG.wrist_rotation_range),
            rng.uniform(-0.5, 0.5)
        ), 'XYZ')
    
    bpy.ops.object.mode_set(mode='OBJECT')


# ============================================================================
# RENDERING
# ============================================================================

def render_to_file(filepath: str):
    """Render current scene to file."""
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def render_to_base64() -> str:
    """Render current scene and return as base64 string."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    bpy.context.scene.render.filepath = tmp_path
    bpy.ops.render.render(write_still=True)
    
    with open(tmp_path, 'rb') as f:
        img_data = f.read()
    
    os.remove(tmp_path)
    
    return base64.b64encode(img_data).decode('utf-8')


def extract_keypoints(armature: bpy.types.Object, camera: bpy.types.Object) -> Dict[str, Any]:
    """Extract 2D and 3D keypoints from the hand armature."""
    from bpy_extras.object_utils import world_to_camera_view
    
    scene = bpy.context.scene
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    
    keypoints_3d = []
    keypoints_2d = []
    bone_names = []
    
    for bone in armature.pose.bones:
        # Get world position of bone head
        world_pos = armature.matrix_world @ bone.head
        keypoints_3d.append([world_pos.x, world_pos.y, world_pos.z])
        
        # Project to 2D camera coordinates
        co_2d = world_to_camera_view(scene, camera, world_pos)
        
        # Convert to pixel coordinates
        px_x = co_2d.x * render_size[0]
        px_y = (1 - co_2d.y) * render_size[1]  # Flip Y axis
        visibility = 1.0 if 0 <= co_2d.x <= 1 and 0 <= co_2d.y <= 1 and co_2d.z > 0 else 0.0
        
        keypoints_2d.append([px_x, px_y, visibility])
        bone_names.append(bone.name)
    
    return {
        'keypoints_3d': keypoints_3d,
        'keypoints_2d': keypoints_2d,
        'bone_names': bone_names
    }


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def generate_dataset():
    """Main function to generate the synthetic ASL dataset."""
    print("=" * 60)
    print("Synthetic ASL Data Generation Pipeline")
    print("=" * 60)
    
    # Set random seed
    if CONFIG.seed is not None:
        random.seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)
    
    # Setup scene
    print("\n[1/5] Setting up scene...")
    clear_scene()
    setup_render_settings()
    camera = create_camera()
    lights = create_lighting()
    
    # Create hand model with armature
    print("[2/5] Creating hand model...")
    hand, armature = create_hand_armature()
    
    # Create output directory
    output_dir = Path(CONFIG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which handshapes to generate
    if CONFIG.target_handshapes:
        handshapes_to_generate = CONFIG.target_handshapes
    else:
        handshapes_to_generate = get_all_handshapes()
    
    print(f"[3/5] Generating {len(handshapes_to_generate)} handshapes, "
          f"{CONFIG.samples_per_handshape} samples each...")
    
    # Metadata storage
    metadata = {
        "dataset_name": "synthetic_asl_slr",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "config": {
            "image_size": CONFIG.image_size,
            "samples_per_handshape": CONFIG.samples_per_handshape,
            "render_engine": CONFIG.render_engine,
            "render_samples": CONFIG.render_samples,
        },
        "handshapes": [],
        "samples": []
    }
    
    sample_id = 0
    total_samples = len(handshapes_to_generate) * CONFIG.samples_per_handshape
    
    # Generate samples for each handshape
    for handshape_name in handshapes_to_generate:
        handshape_config = get_handshape(handshape_name)
        print(f"\n  Generating: {handshape_name} - {handshape_config.description[:40]}...")
        
        # Add handshape info to metadata
        metadata["handshapes"].append({
            "name": handshape_name,
            "description": handshape_config.description,
            "movement_required": handshape_config.movement_required
        })
        
        # Create subdirectory for this handshape
        shape_dir = output_dir / handshape_name
        shape_dir.mkdir(exist_ok=True)
        
        for i in range(CONFIG.samples_per_handshape):
            seed = sample_id
            
            # Randomize skin tone
            skin_idx = seed % len(CONFIG.skin_tones)
            skin_color = CONFIG.skin_tones[skin_idx]
            skin_mat = create_skin_material(skin_color)
            apply_material(hand, skin_mat)
            
            # Apply handshape pose
            apply_handshape_pose(armature, handshape_name, seed)
            
            # Randomize camera
            randomize_camera(camera, seed + 1000)
            
            # Randomize lighting
            randomize_lighting(lights, seed + 2000)
            
            # Randomize background
            bg_type = CONFIG.background_types[seed % len(CONFIG.background_types)]
            set_background(bg_type, seed + 3000)
            
            # Generate filename
            filename = f"{handshape_name}_{i:04d}.png"
            filepath = str(shape_dir / filename)
            
            # Render
            if CONFIG.export_format == "BASE64":
                img_base64 = render_to_base64()
                sample_meta = {
                    "id": sample_id,
                    "handshape": handshape_name,
                    "skin_tone_idx": skin_idx,
                    "seed": seed,
                    "image_base64": img_base64
                }
            else:
                render_to_file(filepath)
                sample_meta = {
                    "id": sample_id,
                    "handshape": handshape_name,
                    "skin_tone_idx": skin_idx,
                    "seed": seed,
                    "filepath": str(Path(handshape_name) / filename)
                }
            
            # Extract and add keypoints
            if CONFIG.export_keypoints:
                keypoints = extract_keypoints(armature, camera)
                sample_meta["keypoints_2d"] = keypoints["keypoints_2d"]
                sample_meta["keypoints_3d"] = keypoints["keypoints_3d"]
                sample_meta["bone_names"] = keypoints["bone_names"]
            
            metadata["samples"].append(sample_meta)
            sample_id += 1
            
            # Progress update
            if sample_id % 50 == 0:
                progress = (sample_id / total_samples) * 100
                print(f"    Progress: {sample_id}/{total_samples} ({progress:.1f}%)")
            
            # Clean up material
            bpy.data.materials.remove(skin_mat)
    
    # Save metadata
    print("\n[4/5] Saving metadata...")
    metadata_path = output_dir / "metadata.json"
    with open(str(metadata_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save label mapping
    label_map = {name: idx for idx, name in enumerate(handshapes_to_generate)}
    label_path = output_dir / "label_map.json"
    with open(str(label_path), 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print("\n[5/5] Generation complete!")
    print(f"  Total samples: {sample_id}")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Label map: {label_path}")


def parse_args():
    """Parse command line arguments."""
    # Find -- separator to get Blender script args
    try:
        idx = sys.argv.index('--')
        args = sys.argv[idx + 1:]
    except ValueError:
        args = []
    
    parser = argparse.ArgumentParser(description='Generate synthetic ASL training data')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--samples', type=int, help='Samples per handshape')
    parser.add_argument('--handshapes', type=str, help='Comma-separated list of handshapes')
    parser.add_argument('--format', type=str, choices=['PNG', 'BASE64'], help='Export format')
    parser.add_argument('--gpu', action='store_true', help='Use GPU rendering')
    parser.add_argument('--engine', type=str, choices=['CYCLES', 'EEVEE'], help='Render engine')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    parsed = parser.parse_args(args)
    
    # Apply to config
    if parsed.output:
        CONFIG.output_dir = parsed.output
    if parsed.samples:
        CONFIG.samples_per_handshape = parsed.samples
    if parsed.handshapes:
        CONFIG.target_handshapes = [h.strip().upper() for h in parsed.handshapes.split(',')]
    if parsed.format:
        CONFIG.export_format = parsed.format
    if parsed.gpu:
        CONFIG.use_gpu = True
    if parsed.engine:
        CONFIG.render_engine = parsed.engine
    if parsed.seed is not None:
        CONFIG.seed = parsed.seed


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parse_args()
    generate_dataset()
