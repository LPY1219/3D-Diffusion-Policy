"""
Check Franka 3-camera dataset format and structure

This script analyzes the format of any Franka task dataset with 3 cameras (e.g., cook5, push_T_5, etc.)
"""
import os
import pickle
import argparse
import numpy as np
from pathlib import Path

def check_data_format(data_root):
    data_root = Path(data_root)

    # Find first trail
    trails = sorted([d for d in os.listdir(data_root) if d.startswith("trail_")])
    if not trails:
        print(f"Error: No trail directories found in {data_root}")
        return

    trail_1 = data_root / trails[0]

    print("="*50)
    print(f"Checking Franka 3-camera dataset format")
    print(f"Dataset: {data_root}")
    print("="*50)

    # Check extrinsics
    print("\n1. Extrinsics:")
    try:
        with open(trail_1 / "extrinsics.pkl", 'rb') as f:
            extrinsics = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"  Error loading with pickle: {e}")
        print("  Trying alternative method...")
        import pickle5
        with open(trail_1 / "extrinsics.pkl", 'rb') as f:
            extrinsics = pickle5.load(f)
    if isinstance(extrinsics, dict):
        print(f"  Type: dict with keys: {list(extrinsics.keys())}")
        for key, value in extrinsics.items():
            print(f"  {key} shape: {np.array(value).shape}")
            print(f"  {key}:\n{np.array(value)}")
    else:
        print(f"  Type: {type(extrinsics)}")
        print(f"  Value:\n{extrinsics}")

    # Check pose
    print("\n2. Pose (first timestep):")
    pose_files = sorted(os.listdir(trail_1 / "poses"))
    with open(trail_1 / "poses" / pose_files[0], 'rb') as f:
        pose = pickle.load(f)
    print(f"  Type: {type(pose)}")
    pose_array = np.array(pose)
    print(f"  Shape: {pose_array.shape}")
    print(f"  Value: {pose_array}")
    print(f"  (Expected: 7 values - xyz + quaternion wxyz)")

    # Check gripper
    print("\n3. Gripper state (first timestep):")
    gripper_files = sorted(os.listdir(trail_1 / "gripper_states"))
    with open(trail_1 / "gripper_states" / gripper_files[0], 'rb') as f:
        gripper = pickle.load(f)
    print(f"  Type: {type(gripper)}")
    print(f"  Value: {gripper}")

    # Check point cloud from camera 1
    print("\n4. Point cloud (camera 1, first timestep):")
    pcd_files = sorted(os.listdir(trail_1 / "3rd_1_pcd"))
    with open(trail_1 / "3rd_1_pcd" / pcd_files[0], 'rb') as f:
        pcd = pickle.load(f)
    print(f"  Type: {type(pcd)}")
    pcd_array = np.array(pcd)
    print(f"  Shape: {pcd_array.shape}")
    print(f"  Sample points (first 3):\n{pcd_array[:3]}")
    print(f"  Min values: {np.min(pcd_array, axis=0)}")
    print(f"  Max values: {np.max(pcd_array, axis=0)}")

    # Check RGB image
    print("\n5. RGB image (camera 1, first timestep):")
    rgb_files = sorted(os.listdir(trail_1 / "3rd_1_bgr"))
    with open(trail_1 / "3rd_1_bgr" / rgb_files[0], 'rb') as f:
        rgb = pickle.load(f)
    print(f"  Type: {type(rgb)}")
    rgb_array = np.array(rgb)
    print(f"  Shape: {rgb_array.shape}")

    # Check depth
    print("\n6. Depth image (camera 1, first timestep):")
    depth_files = sorted(os.listdir(trail_1 / "3rd_1_depth"))
    with open(trail_1 / "3rd_1_depth" / depth_files[0], 'rb') as f:
        depth = pickle.load(f)
    print(f"  Type: {type(depth)}")
    depth_array = np.array(depth)
    print(f"  Shape: {depth_array.shape}")

    # Check number of timesteps
    print("\n7. Number of timesteps:")
    print(f"  Poses: {len(pose_files)}")
    print(f"  Grippers: {len(gripper_files)}")
    print(f"  Camera 1 PCDs: {len(pcd_files)}")

    # Check all trails
    print("\n8. All trails:")
    all_trails = sorted([d for d in os.listdir(data_root) if d.startswith("trail_")])
    print(f"  Number of trails: {len(all_trails)}")
    for trail in all_trails:
        trail_path = data_root / trail
        n_steps = len(os.listdir(trail_path / "poses"))
        print(f"  {trail}: {n_steps} timesteps")

    print("\n" + "="*50)
    print("Format check complete!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Check Franka 3-camera dataset format and structure'
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to raw dataset directory (e.g., /path/to/cook_5 or /path/to/push_T_5)')

    args = parser.parse_args()
    check_data_format(args.data_root)
