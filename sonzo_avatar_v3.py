#!/usr/bin/env python3
"""
SonZo Avatar Engine v3 - Proper SMPL-X Posing
Uses SMPL-X pose parameters directly for correct body deformation
"""

import bpy
import math
import sys
import os
import argparse

# Add path for external modules
sys.path.insert(0, '/home/ubuntu/.local/lib/python3.12/site-packages')
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import numpy as np

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
    print("✅ SMPL-X loaded successfully")
except ImportError as e:
    SMPLX_AVAILABLE = False
    print(f"⚠️ SMPL-X not available: {e}")

SMPLX_MODEL_PATH = '/home/ubuntu/sonzo_avatar/models/models'

# SMPL-X joint indices (55 joints total for body_pose, 3 values each = 165 params)
# Key joints for signing:
# 0: pelvis, 1-2: left/right hip, 3-5: spine, 6-8: legs, 9: spine2
# 12: neck, 15: head
# 16: left_shoulder, 17: left_elbow, 18: left_wrist
# 17: right_shoulder, 18: right_elbow, 19: right_wrist

# Sign animations using SMPL-X pose parameters
# Each keyframe has body_pose adjustments (joint_index: [rx, ry, rz])
SIGN_ANIMATIONS = {
    'HELLO': {
        'description': 'Wave hello',
        'frames': 30,
        'keyframes': [
            {
                'frame': 0,
                'joints': {
                    16: [0, 0, -0.5],      # right shoulder out
                    17: [-0.8, 0, 0],      # right elbow bent
                }
            },
            {
                'frame': 10,
                'joints': {
                    16: [0, 0.3, -0.5],    # right shoulder wave right
                    17: [-0.8, 0, 0],
                }
            },
            {
                'frame': 20,
                'joints': {
                    16: [0, -0.3, -0.5],   # right shoulder wave left
                    17: [-0.8, 0, 0],
                }
            },
            {
                'frame': 30,
                'joints': {
                    16: [0, 0, -0.5],      # back to center
                    17: [-0.8, 0, 0],
                }
            },
        ]
    },
    'THANK_YOU': {
        'description': 'Thank you - hand from chin forward',
        'frames': 24,
        'keyframes': [
            {
                'frame': 0,
                'joints': {
                    16: [0.3, 0, -0.3],
                    17: [-1.2, 0, 0],
                }
            },
            {
                'frame': 12,
                'joints': {
                    16: [0.5, 0, -0.5],
                    17: [-0.8, 0, 0],
                }
            },
            {
                'frame': 24,
                'joints': {
                    16: [0.3, 0, -0.7],
                    17: [-0.5, 0, 0],
                }
            },
        ]
    },
    'YES': {
        'description': 'Yes - fist nod',
        'frames': 24,
        'keyframes': [
            {
                'frame': 0,
                'joints': {
                    16: [0.2, 0, -0.3],
                    17: [-0.9, 0, 0],
                }
            },
            {
                'frame': 8,
                'joints': {
                    16: [0.4, 0, -0.3],
                    17: [-0.7, 0, 0],
                }
            },
            {
                'frame': 16,
                'joints': {
                    16: [0.2, 0, -0.3],
                    17: [-0.9, 0, 0],
                }
            },
            {
                'frame': 24,
                'joints': {
                    16: [0.4, 0, -0.3],
                    17: [-0.7, 0, 0],
                }
            },
        ]
    },
    'NO': {
        'description': 'No - finger tap',
        'frames': 24,
        'keyframes': [
            {
                'frame': 0,
                'joints': {
                    16: [0.2, 0, -0.4],
                    17: [-1.0, 0, 0],
                }
            },
            {
                'frame': 12,
                'joints': {
                    16: [0.3, 0, -0.4],
                    17: [-0.8, 0, 0],
                }
            },
            {
                'frame': 24,
                'joints': {
                    16: [0.2, 0, -0.4],
                    17: [-1.0, 0, 0],
                }
            },
        ]
    },
    'I_LOVE_YOU': {
        'description': 'ILY handshape',
        'frames': 30,
        'keyframes': [
            {
                'frame': 0,
                'joints': {
                    16: [0, 0, 0],
                    17: [0, 0, 0],
                }
            },
            {
                'frame': 15,
                'joints': {
                    16: [0, 0.2, -0.8],
                    17: [-0.5, 0, 0],
                }
            },
            {
                'frame': 30,
                'joints': {
                    16: [0, 0.2, -0.8],
                    17: [-0.5, 0, 0],
                }
            },
        ]
    },
}


def clear_scene():
    """Remove all objects from scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def interpolate_keyframes(keyframes, frame, num_joints=22):
    """Interpolate between keyframes to get pose at specific frame"""
    # Find surrounding keyframes
    prev_kf = keyframes[0]
    next_kf = keyframes[-1]
    
    for i, kf in enumerate(keyframes):
        if kf['frame'] <= frame:
            prev_kf = kf
        if kf['frame'] >= frame and (next_kf['frame'] < frame or kf['frame'] < next_kf['frame']):
            next_kf = kf
            break
    
    # Calculate interpolation factor
    if prev_kf['frame'] == next_kf['frame']:
        t = 0
    else:
        t = (frame - prev_kf['frame']) / (next_kf['frame'] - prev_kf['frame'])
    
    # Smooth interpolation
    t = t * t * (3 - 2 * t)  # smoothstep
    
    # Create pose array (22 joints * 3 = 66 values for body_pose)
    pose = np.zeros((num_joints, 3))
    
    # Interpolate each joint
    all_joints = set(prev_kf.get('joints', {}).keys()) | set(next_kf.get('joints', {}).keys())
    
    for joint_idx in all_joints:
        prev_val = np.array(prev_kf.get('joints', {}).get(joint_idx, [0, 0, 0]))
        next_val = np.array(next_kf.get('joints', {}).get(joint_idx, [0, 0, 0]))
        
        if joint_idx < num_joints:
            pose[joint_idx] = prev_val * (1 - t) + next_val * t
    
    return pose.flatten()


def create_smplx_mesh_for_frame(model, body_pose):
    """Generate SMPL-X mesh for a specific pose"""
    body_pose_tensor = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0)
    
    # Generate mesh
    output = model(body_pose=body_pose_tensor)
    vertices = output.vertices.detach().numpy()[0]
    
    return vertices


def setup_scene():
    """Set up lighting and camera"""
    # Camera - portrait orientation, framing upper body
    bpy.ops.object.camera_add(location=(0, -2.0, 0.3))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(90), 0, 0)
    camera.name = "Camera"
    bpy.context.scene.camera = camera
    
    # Adjust camera to frame upper body
    camera.data.lens = 50
    
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
    
    # Background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.15, 0.15, 0.18, 1.0)


def create_material():
    """Create skin material"""
    mat = bpy.data.materials.new(name="Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.6, 0.5, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.5
    return mat


def render_animation(model, sign_name, output_path):
    """Render sign animation by generating mesh for each frame"""
    if sign_name not in SIGN_ANIMATIONS:
        print(f"Unknown sign: {sign_name}")
        return
    
    sign = SIGN_ANIMATIONS[sign_name]
    total_frames = sign['frames']
    keyframes = sign['keyframes']
    
    # Set frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames
    
    # Create material once
    skin_mat = create_material()
    
    # Get face topology (constant)
    faces = model.faces
    
    # Create initial mesh object
    mesh = bpy.data.meshes.new('SMPLX_Body')
    avatar = bpy.data.objects.new('Avatar', mesh)
    bpy.context.collection.objects.link(avatar)
    
    # Get initial pose vertices
    init_pose = interpolate_keyframes(keyframes, 0)
    init_verts = create_smplx_mesh_for_frame(model, init_pose)
    mesh.from_pydata(init_verts.tolist(), [], faces.tolist())
    mesh.update()
    
    # Smooth shading
    for poly in mesh.polygons:
        poly.use_smooth = True
    
    # Apply material
    avatar.data.materials.append(skin_mat)
    
    # Add shape keys for animation
    avatar.shape_key_add(name='Basis', from_mix=False)
    
    # Create shape key for each frame
    print(f"Generating {total_frames + 1} frames...")
    for frame in range(total_frames + 1):
        pose = interpolate_keyframes(keyframes, frame)
        verts = create_smplx_mesh_for_frame(model, pose)
        
        # Add shape key
        sk = avatar.shape_key_add(name=f'Frame_{frame}', from_mix=False)
        for i, v in enumerate(sk.data):
            v.co = verts[i]
        
        # Keyframe the shape key
        sk.value = 0
        sk.keyframe_insert(data_path='value', frame=max(0, frame - 1))
        sk.value = 1
        sk.keyframe_insert(data_path='value', frame=frame)
        sk.value = 0
        sk.keyframe_insert(data_path='value', frame=min(total_frames, frame + 1))
        
        if frame % 5 == 0:
            print(f"  Frame {frame}/{total_frames}")
    
    # Set up rendering
    output_file = os.path.join(output_path, f"{sign_name.lower()}.mp4")
    bpy.context.scene.render.filepath = output_file
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
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
    print("SonZo Avatar Engine v3 - SMPL-X Direct Posing")
    print("=" * 50)
    
    # Parse arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sign', type=str, help='Sign to animate')
    parser.add_argument('--output', type=str, default='/home/ubuntu/sonzo_avatar/output')
    parser.add_argument('--list-signs', action='store_true')
    
    args = parser.parse_args(argv)
    
    if args.list_signs:
        print("\nAvailable signs:")
        for name, data in SIGN_ANIMATIONS.items():
            print(f"  - {name}: {data['description']}")
        return
    
    if not args.sign:
        print("Use --sign NAME or --list-signs")
        return
    
    sign_name = args.sign.upper()
    
    if not SMPLX_AVAILABLE:
        print("SMPL-X not available!")
        return
    
    # Load SMPL-X model
    print("Loading SMPL-X model...")
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10
    )
    
    # Clear and setup scene
    clear_scene()
    setup_scene()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Render animation
    render_animation(model, sign_name, args.output)
    
    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
