#!/usr/bin/env python3
"""
SonZo Avatar Engine v2 - SMPL-X Integration
Generates realistic signing avatars using SMPL-X body model
"""

import bpy
import bmesh
import math
import sys
import os
import argparse

# Add path for external modules
sys.path.insert(0, '/home/ubuntu/.local/lib/python3.12/site-packages')
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import numpy as np

# Try to import smplx and torch
try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
    print("✅ SMPL-X loaded successfully")
except ImportError as e:
    SMPLX_AVAILABLE = False
    print(f"⚠️ SMPL-X not available: {e}")

# SMPL-X model path
SMPLX_MODEL_PATH = '/home/ubuntu/sonzo_avatar/models/models'

# Sign animation definitions (hand poses and arm positions)
# Values are approximate joint angles in radians
SIGN_ANIMATIONS = {
    'HELLO': {
        'description': 'Wave hello - open hand waves side to side',
        'frames': 30,
        'keyframes': [
            {'frame': 0, 'right_arm': [0, 0, -1.2], 'right_hand': 'open', 'left_arm': [0, 0, 0]},
            {'frame': 10, 'right_arm': [0, 0.3, -1.2], 'right_hand': 'open', 'left_arm': [0, 0, 0]},
            {'frame': 20, 'right_arm': [0, -0.3, -1.2], 'right_hand': 'open', 'left_arm': [0, 0, 0]},
            {'frame': 30, 'right_arm': [0, 0, -1.2], 'right_hand': 'open', 'left_arm': [0, 0, 0]},
        ]
    },
    'THANK_YOU': {
        'description': 'Thank you - flat hand from chin outward',
        'frames': 24,
        'keyframes': [
            {'frame': 0, 'right_arm': [0.5, 0, -0.8], 'right_hand': 'flat', 'left_arm': [0, 0, 0]},
            {'frame': 12, 'right_arm': [0.3, 0, -1.0], 'right_hand': 'flat', 'left_arm': [0, 0, 0]},
            {'frame': 24, 'right_arm': [0.1, 0, -0.6], 'right_hand': 'flat', 'left_arm': [0, 0, 0]},
        ]
    },
    'YES': {
        'description': 'Yes - fist nodding',
        'frames': 24,
        'keyframes': [
            {'frame': 0, 'right_arm': [0.3, 0, -0.5], 'right_hand': 'fist', 'left_arm': [0, 0, 0]},
            {'frame': 8, 'right_arm': [0.5, 0, -0.5], 'right_hand': 'fist', 'left_arm': [0, 0, 0]},
            {'frame': 16, 'right_arm': [0.3, 0, -0.5], 'right_hand': 'fist', 'left_arm': [0, 0, 0]},
            {'frame': 24, 'right_arm': [0.5, 0, -0.5], 'right_hand': 'fist', 'left_arm': [0, 0, 0]},
        ]
    },
    'NO': {
        'description': 'No - index and middle finger tap thumb',
        'frames': 24,
        'keyframes': [
            {'frame': 0, 'right_arm': [0.3, 0, -0.7], 'right_hand': 'no_open', 'left_arm': [0, 0, 0]},
            {'frame': 8, 'right_arm': [0.3, 0, -0.7], 'right_hand': 'no_closed', 'left_arm': [0, 0, 0]},
            {'frame': 16, 'right_arm': [0.3, 0, -0.7], 'right_hand': 'no_open', 'left_arm': [0, 0, 0]},
            {'frame': 24, 'right_arm': [0.3, 0, -0.7], 'right_hand': 'no_closed', 'left_arm': [0, 0, 0]},
        ]
    },
    'I_LOVE_YOU': {
        'description': 'I love you - ILY handshape',
        'frames': 30,
        'keyframes': [
            {'frame': 0, 'right_arm': [0, 0, 0], 'right_hand': 'fist', 'left_arm': [0, 0, 0]},
            {'frame': 15, 'right_arm': [0.2, 0.3, -1.0], 'right_hand': 'ily', 'left_arm': [0, 0, 0]},
            {'frame': 30, 'right_arm': [0.2, 0.3, -1.0], 'right_hand': 'ily', 'left_arm': [0, 0, 0]},
        ]
    },
}

def clear_scene():
    """Remove all objects from scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_smplx_mesh():
    """Create mesh from SMPL-X model"""
    if not SMPLX_AVAILABLE:
        print("SMPL-X not available, using fallback")
        return create_fallback_avatar()
    
    # Load SMPL-X model
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True
    )
    
    # Get default pose vertices
    output = model()
    vertices = output.vertices.detach().numpy()[0]
    faces = model.faces
    
    # Create mesh in Blender
    mesh = bpy.data.meshes.new('SMPLX_Body')
    obj = bpy.data.objects.new('Avatar', mesh)
    
    # Link to scene
    bpy.context.collection.objects.link(obj)
    
    # Create mesh from vertices and faces
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    
    # Add smooth shading
    for poly in mesh.polygons:
        poly.use_smooth = True
    
    # Create and assign skin material
    mat = bpy.data.materials.new(name="Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)  # Skin tone
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Subsurface Weight'].default_value = 0.3
        bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.5, 0.3)
    
    obj.data.materials.append(mat)
    
    return obj, model

def create_fallback_avatar():
    """Create simple avatar if SMPL-X unavailable"""
    # Create body (torso)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
    torso = bpy.context.active_object
    torso.scale = (0.4, 0.25, 0.6)
    torso.name = "Avatar"
    
    # Create head
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 1.1))
    head = bpy.context.active_object
    head.name = "Head"
    head.parent = torso
    
    # Create arms
    for side, x_pos in [('Right', 0.5), ('Left', -0.5)]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.6, location=(x_pos, 0, 0.5))
        arm = bpy.context.active_object
        arm.rotation_euler = (0, math.radians(90), 0)
        arm.name = f"{side}Arm"
        arm.parent = torso
    
    # Add material
    mat = bpy.data.materials.new(name="Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
    torso.data.materials.append(mat)
    
    return torso, None

def create_armature_for_smplx(mesh_obj):
    """Create armature for SMPL-X mesh animation"""
    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = "AvatarArmature"
    
    # Get armature data
    arm_data = armature.data
    arm_data.name = "AvatarRig"
    
    # Clear default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    # Create spine bones
    bones_data = [
        ('pelvis', (0, 0, 0), (0, 0, 0.2)),
        ('spine', (0, 0, 0.2), (0, 0, 0.4)),
        ('spine1', (0, 0, 0.4), (0, 0, 0.55)),
        ('spine2', (0, 0, 0.55), (0, 0, 0.7)),
        ('neck', (0, 0, 0.7), (0, 0, 0.8)),
        ('head', (0, 0, 0.8), (0, 0, 1.0)),
        # Right arm
        ('right_collar', (0, 0, 0.65), (0.15, 0, 0.65)),
        ('right_shoulder', (0.15, 0, 0.65), (0.35, 0, 0.55)),
        ('right_elbow', (0.35, 0, 0.55), (0.55, 0, 0.45)),
        ('right_wrist', (0.55, 0, 0.45), (0.65, 0, 0.4)),
        # Left arm
        ('left_collar', (0, 0, 0.65), (-0.15, 0, 0.65)),
        ('left_shoulder', (-0.15, 0, 0.65), (-0.35, 0, 0.55)),
        ('left_elbow', (-0.35, 0, 0.55), (-0.55, 0, 0.45)),
        ('left_wrist', (-0.55, 0, 0.45), (-0.65, 0, 0.4)),
    ]
    
    for bone_name, head, tail in bones_data:
        bone = arm_data.edit_bones.new(bone_name)
        bone.head = head
        bone.tail = tail
    
    # Set up bone parenting
    arm_data.edit_bones['spine'].parent = arm_data.edit_bones['pelvis']
    arm_data.edit_bones['spine1'].parent = arm_data.edit_bones['spine']
    arm_data.edit_bones['spine2'].parent = arm_data.edit_bones['spine1']
    arm_data.edit_bones['neck'].parent = arm_data.edit_bones['spine2']
    arm_data.edit_bones['head'].parent = arm_data.edit_bones['neck']
    
    arm_data.edit_bones['right_collar'].parent = arm_data.edit_bones['spine2']
    arm_data.edit_bones['right_shoulder'].parent = arm_data.edit_bones['right_collar']
    arm_data.edit_bones['right_elbow'].parent = arm_data.edit_bones['right_shoulder']
    arm_data.edit_bones['right_wrist'].parent = arm_data.edit_bones['right_elbow']
    
    arm_data.edit_bones['left_collar'].parent = arm_data.edit_bones['spine2']
    arm_data.edit_bones['left_shoulder'].parent = arm_data.edit_bones['left_collar']
    arm_data.edit_bones['left_elbow'].parent = arm_data.edit_bones['left_shoulder']
    arm_data.edit_bones['left_wrist'].parent = arm_data.edit_bones['left_elbow']
    
    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Parent mesh to armature with automatic weights
    mesh_obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    
    return armature

def setup_scene():
    """Set up lighting and camera for portrait video"""
    # Camera - portrait orientation, upper body framing
    bpy.ops.object.camera_add(location=(0, -2.5, 0.6))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(90), 0, 0)
    camera.name = "AvatarCamera"
    bpy.context.scene.camera = camera
    
    # Set render resolution (portrait 9:16)
    bpy.context.scene.render.resolution_x = 720
    bpy.context.scene.render.resolution_y = 1280
    
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(2, -2, 2))
    key_light = bpy.context.active_object
    key_light.data.energy = 500
    key_light.data.size = 2
    key_light.name = "KeyLight"
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, -1, 1))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 200
    fill_light.data.size = 3
    fill_light.name = "FillLight"
    
    # Rim light
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 300
    rim_light.data.size = 2
    rim_light.name = "RimLight"
    
    # Background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.15, 0.15, 0.18, 1.0)  # Dark gray

def animate_sign(armature, sign_name):
    """Apply sign animation to armature"""
    if sign_name not in SIGN_ANIMATIONS:
        print(f"Sign '{sign_name}' not found")
        return
    
    sign = SIGN_ANIMATIONS[sign_name]
    total_frames = sign['frames']
    
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames
    
    # Go to pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    for keyframe in sign['keyframes']:
        frame = keyframe['frame']
        bpy.context.scene.frame_set(frame)
        
        # Animate right arm
        if 'right_arm' in keyframe:
            angles = keyframe['right_arm']
            
            # Shoulder rotation
            if 'right_shoulder' in armature.pose.bones:
                bone = armature.pose.bones['right_shoulder']
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = (angles[0], angles[1], angles[2])
                bone.keyframe_insert(data_path='rotation_euler', frame=frame)
            
            # Elbow bend
            if 'right_elbow' in armature.pose.bones:
                bone = armature.pose.bones['right_elbow']
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = (angles[0] * 0.5, 0, 0)
                bone.keyframe_insert(data_path='rotation_euler', frame=frame)
        
        # Animate left arm
        if 'left_arm' in keyframe:
            angles = keyframe['left_arm']
            if 'left_shoulder' in armature.pose.bones:
                bone = armature.pose.bones['left_shoulder']
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = (angles[0], -angles[1], -angles[2])
                bone.keyframe_insert(data_path='rotation_euler', frame=frame)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Applied animation for '{sign_name}' ({total_frames} frames)")

def render_video(output_path, sign_name):
    """Render animation to video file"""
    # Set output path
    output_file = os.path.join(output_path, f"{sign_name.lower()}.mp4")
    bpy.context.scene.render.filepath = output_file
    
    # Use EEVEE for faster rendering (no denoiser needed)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    
    # EEVEE settings for quality
    bpy.context.scene.eevee.taa_render_samples = 64
    
    # Disable denoiser (not needed for EEVEE)
    bpy.context.scene.view_layers[0].use_pass_denoising_data = False
    
    # Output format
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    bpy.context.scene.render.fps = 24
    
    print(f"Rendering to: {output_file}")
    bpy.ops.render.render(animation=True)
    print(f"✅ Generated: {output_file}")
    
    return output_file

def main():
    print("=" * 50)
    print("SonZo Avatar Engine v2 - SMPL-X Integration")
    print("=" * 50)
    
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description='Generate signing avatar videos')
    parser.add_argument('--sign', type=str, help='Sign to animate (e.g., HELLO, THANK_YOU)')
    parser.add_argument('--output', type=str, default='/home/ubuntu/sonzo_avatar/output', help='Output directory')
    parser.add_argument('--list-signs', action='store_true', help='List available signs')
    
    args = parser.parse_args(argv)
    
    if args.list_signs:
        print("\nAvailable signs:")
        for sign_name, sign_data in SIGN_ANIMATIONS.items():
            print(f"  - {sign_name}: {sign_data['description']}")
        return
    
    if not args.sign:
        print("No sign specified. Use --sign HELLO or --list-signs")
        return
    
    sign_name = args.sign.upper()
    if sign_name not in SIGN_ANIMATIONS:
        print(f"Unknown sign: {sign_name}")
        print("Use --list-signs to see available signs")
        return
    
    # Clear and set up scene
    clear_scene()
    
    # Create SMPL-X avatar
    print("Creating SMPL-X avatar...")
    mesh_obj, smplx_model = create_smplx_mesh()
    
    # Create armature and rig
    print("Creating armature...")
    armature = create_armature_for_smplx(mesh_obj)
    
    # Set up scene (camera, lights)
    print("Setting up scene...")
    setup_scene()
    
    # Apply sign animation
    print(f"Animating sign: {sign_name}")
    animate_sign(armature, sign_name)
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Render video
    print("Rendering video...")
    render_video(args.output, sign_name)
    
    print("=" * 50)
    print("Done!")

if __name__ == "__main__":
    main()
