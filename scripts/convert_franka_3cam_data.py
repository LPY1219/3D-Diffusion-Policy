"""
Convert Franka 3-camera dataset to zarr format for 3D-Diffusion-Policy training

This script supports any Franka task with 3 cameras (e.g., cook5, push_T_5, etc.)
"""
import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
from termcolor import cprint
from pathlib import Path


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """
    FPS sampling to downsample point cloud
    Args:
        points: (N, 3) numpy array
        num_points: target number of points
    Returns:
        sampled_points: (num_points, 3) numpy array
    """
    K = [num_points]
    if use_cuda and torch.cuda.is_available():
        points_tensor = torch.from_numpy(points).cuda().float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).cpu().numpy()
        indices = indices.cpu()
    else:
        points_tensor = torch.from_numpy(points).float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).numpy()

    return sampled_points, indices


def convert_pcd_to_base(extrinsic_matrix, pcd):
    """
    Transform point cloud from camera frame to robot base frame using extrinsic matrix

    Reference: BridgeVLA_dev/Wan/DiffSynth-Studio/diffsynth/trainers/
               base_multi_view_dataset_with_rot_grip_3cam_different_projection.py

    Args:
        extrinsic_matrix: (4, 4) transformation matrix from camera to robot base
        pcd: (H, W, 3) or (N, 3) numpy array of XYZ coordinates in camera frame
    Returns:
        transformed_pcd: same shape as input, XYZ coordinates in robot base frame
    """
    original_shape = pcd.shape

    # Reshape to (N, 3)
    pcd_flat = pcd.reshape(-1, 3)

    # Create homogeneous coordinates (N, 4)
    ones = np.ones((pcd_flat.shape[0], 1))
    pcd_homo = np.concatenate((pcd_flat, ones), axis=1)

    # Apply transformation: transform from camera frame to base frame
    pcd_transformed = (extrinsic_matrix @ pcd_homo.T).T[:, :3]

    # Reshape back to original shape
    return pcd_transformed.reshape(original_shape)


def crop_point_cloud(points, scene_bounds):
    """
    Crop point cloud to scene bounds
    Args:
        points: (N, 3) numpy array
        scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        cropped_points: (M, 3) numpy array where M <= N
    """
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds

    mask = (
        (points[:, 0] > x_min) & (points[:, 0] < x_max) &
        (points[:, 1] > y_min) & (points[:, 1] < y_max) &
        (points[:, 2] > z_min) & (points[:, 2] < z_max)
    )

    return points[mask]


def process_point_cloud(pcd_files, extrinsics, scene_bounds, num_points=1024, use_cuda=True):
    """
    Process point clouds from 3 cameras:
    1. Load point clouds from 3 cameras
    2. Transform each to robot base frame using extrinsics
    3. Merge point clouds
    4. Remove invalid points (0, 0, 0)
    5. Crop to scene bounds
    6. FPS downsample to num_points

    Args:
        pcd_files: dict with keys '3rd_1', '3rd_2', '3rd_3', values are file paths
        extrinsics: dict with keys '3rd_1', '3rd_2', '3rd_3', values are 4x4 matrices
        scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points: target number of points after FPS
    Returns:
        final_pcd: (num_points, 3) numpy array
    """
    merged_points = []

    for cam_id in ['3rd_1', '3rd_2', '3rd_3']:
        # Load point cloud
        with open(pcd_files[cam_id], 'rb') as f:
            pcd = pickle.load(f)  # Shape: (H, W, 3)

        # Transform to robot base frame using extrinsics
        transformed = convert_pcd_to_base(extrinsics[cam_id], pcd)

        # Reshape to (N, 3)
        transformed_flat = transformed.reshape(-1, 3)

        # Remove invalid points (0, 0, 0) or very close to zero
        valid_mask = np.linalg.norm(transformed_flat, axis=1) > 0.01
        valid_points = transformed_flat[valid_mask]

        merged_points.append(valid_points)

    # Merge all cameras
    merged_points = np.vstack(merged_points)  # (N_total, 3)

    # Crop to scene bounds
    cropped_points = crop_point_cloud(merged_points, scene_bounds)

    # Check if we have enough points
    if len(cropped_points) < num_points:
        cprint(f"Warning: Only {len(cropped_points)} points after cropping, less than {num_points}", 'yellow')
        # Pad with zeros if not enough points
        pad_size = num_points - len(cropped_points)
        padding = np.zeros((pad_size, 3))
        cropped_points = np.vstack([cropped_points, padding])
        return cropped_points.astype(np.float32)

    # FPS downsampling
    final_pcd, _ = farthest_point_sampling(cropped_points, num_points, use_cuda)

    return final_pcd.astype(np.float32)


def quaternion_to_action(curr_pose, prev_pose):
    """
    Compute relative action from previous pose to current pose
    Args:
        curr_pose: (8,) array [x, y, z, qw, qx, qy, qz, gripper]
        prev_pose: (8,) array [x, y, z, qw, qx, qy, qz, gripper]
    Returns:
        action: (8,) array of relative changes
    """
    # Position delta
    pos_delta = curr_pose[:3] - prev_pose[:3]

    # Quaternion delta (simple difference, could use quaternion math for better results)
    quat_delta = curr_pose[3:7] - prev_pose[3:7]

    # Gripper delta
    gripper_delta = curr_pose[7:8] - prev_pose[7:8]

    action = np.concatenate([pos_delta, quat_delta, gripper_delta])
    return action


def convert_franka_3cam_dataset(
    data_root,
    save_path,
    scene_bounds=None,
    num_points=1024,
    use_cuda=True
):
    """
    Convert Franka 3-camera dataset to zarr format

    This function processes any Franka task dataset with 3 cameras following the standard format:
    - 3 camera point clouds (3rd_1_pcd, 3rd_2_pcd, 3rd_3_pcd)
    - Extrinsics file (extrinsics.pkl) for camera-to-base transformations
    - End-effector poses (poses/*.pkl) in [x, y, z, qw, qx, qy, qz] format
    - Gripper states (gripper_states/*.pkl) as boolean values
    - Data organized in trail_X directories

    Args:
        data_root: Path to raw dataset (e.g., cook_5, push_T_5)
        save_path: Path to save zarr file
        scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] for cropping
        num_points: Number of points after FPS downsampling
        use_cuda: Whether to use CUDA for FPS
    """
    if scene_bounds is None:
        # Default scene bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
        scene_bounds = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9]

    cprint(f"Converting Franka 3-camera dataset from {data_root}", 'green')
    cprint(f"Scene bounds: {scene_bounds}", 'green')
    cprint(f"Target number of points: {num_points}", 'green')
    cprint(f"Save path: {save_path}", 'green')

    data_root = Path(data_root)
    trails = sorted([d for d in os.listdir(data_root) if d.startswith("trail_")])

    # Storage
    total_count = 0
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    # Check if save path exists
    if os.path.exists(save_path):
        cprint(f'Data already exists at {save_path}', 'red')
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = input()
        if user_input.lower() == 'y':
            cprint(f'Overwriting {save_path}', 'red')
            os.system(f'rm -rf {save_path}')
        else:
            cprint('Exiting', 'red')
            return

    # Process each trail
    for trail_name in trails:
        trail_path = data_root / trail_name
        cprint(f'\nProcessing {trail_name}', 'cyan')

        # Load extrinsics (same for all timesteps in a trail)
        with open(trail_path / "extrinsics.pkl", 'rb') as f:
            extrinsics = pickle.load(f)

        # Get timestep files
        pose_files = sorted(os.listdir(trail_path / "poses"))
        gripper_files = sorted(os.listdir(trail_path / "gripper_states"))
        pcd_1_files = sorted(os.listdir(trail_path / "3rd_1_pcd"))
        pcd_2_files = sorted(os.listdir(trail_path / "3rd_2_pcd"))
        pcd_3_files = sorted(os.listdir(trail_path / "3rd_3_pcd"))

        num_timesteps = len(pose_files)
        assert len(gripper_files) == num_timesteps, "Mismatch in number of timesteps"
        assert len(pcd_1_files) == num_timesteps, "Mismatch in number of timesteps"

        # Storage for this episode
        episode_point_clouds = []
        episode_states = []

        # Process each timestep
        for step_idx in tqdm.tqdm(range(num_timesteps), desc=f"{trail_name}"):
            # Load pose
            with open(trail_path / "poses" / pose_files[step_idx], 'rb') as f:
                pose = pickle.load(f)  # (7,) [x, y, z, qw, qx, qy, qz]

            # Load gripper state
            with open(trail_path / "gripper_states" / gripper_files[step_idx], 'rb') as f:
                gripper = pickle.load(f)  # bool

            # Convert gripper to float (0.0 or 1.0)
            gripper_value = float(gripper)

            # Combine into state: [x, y, z, qw, qx, qy, qz, gripper]
            state = np.concatenate([pose, [gripper_value]]).astype(np.float32)

            # Process point clouds from 3 cameras
            pcd_files_dict = {
                '3rd_1': trail_path / "3rd_1_pcd" / pcd_1_files[step_idx],
                '3rd_2': trail_path / "3rd_2_pcd" / pcd_2_files[step_idx],
                '3rd_3': trail_path / "3rd_3_pcd" / pcd_3_files[step_idx],
            }

            try:
                point_cloud = process_point_cloud(
                    pcd_files_dict, extrinsics, scene_bounds, num_points, use_cuda
                )
            except Exception as e:
                cprint(f"Error processing point cloud at {trail_name}/{step_idx}: {e}", 'red')
                # Use zero point cloud as fallback
                point_cloud = np.zeros((num_points, 3), dtype=np.float32)

            episode_point_clouds.append(point_cloud)
            episode_states.append(state)
            total_count += 1

        # Compute actions (relative changes)
        episode_actions = []
        for i in range(num_timesteps):
            if i == 0:
                # First action is zero (no previous state)
                action = np.zeros(8, dtype=np.float32)
            else:
                action = quaternion_to_action(episode_states[i], episode_states[i-1])
            episode_actions.append(action)

        # Add to global storage
        point_cloud_arrays.extend(episode_point_clouds)
        state_arrays.extend(episode_states)
        action_arrays.extend(episode_actions)
        episode_ends_arrays.append(total_count)

        cprint(f"  {trail_name}: {num_timesteps} timesteps processed", 'green')

    # Convert to numpy arrays
    point_cloud_arrays = np.array(point_cloud_arrays, dtype=np.float32)
    state_arrays = np.array(state_arrays, dtype=np.float32)
    action_arrays = np.array(action_arrays, dtype=np.float32)
    episode_ends_arrays = np.array(episode_ends_arrays, dtype=np.int64)

    # Create zarr file
    cprint('\nCreating zarr file...', 'cyan')
    zarr_root = zarr.group(save_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Create datasets
    point_cloud_chunk_size = (100, num_points, 3)
    state_chunk_size = (100, 8)
    action_chunk_size = (100, 8)

    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays,
                             chunks=point_cloud_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays,
                             chunks=state_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays,
                             chunks=action_chunk_size, dtype='float32',
                             overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays,
                             chunks=(100,), dtype='int64',
                             overwrite=True, compressor=compressor)

    # Print statistics
    cprint('\n' + '='*50, 'green')
    cprint('Conversion complete!', 'green')
    cprint('='*50, 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays):.3f}, {np.max(point_cloud_arrays):.3f}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays):.3f}, {np.max(state_arrays):.3f}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays):.3f}, {np.max(action_arrays):.3f}]', 'green')
    cprint(f'episode_ends shape: {episode_ends_arrays.shape}, values: {episode_ends_arrays}', 'green')
    cprint(f'Total timesteps: {total_count}', 'green')
    cprint(f'Total episodes: {len(episode_ends_arrays)}', 'green')
    cprint(f'Saved to: {save_path}', 'green')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert Franka 3-camera dataset to zarr format for 3D-Diffusion-Policy training'
    )
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to raw dataset directory (e.g., /path/to/cook_5 or /path/to/push_T_5)')
    parser.add_argument('--save_path', type=str, required=True,
                       help='Path to save zarr file (e.g., /path/to/cook5.zarr)')
    parser.add_argument('--scene_bounds', type=str,
                       default="0.05,-0.5,0,0.85,0.5,0.8",
                       help='Scene bounds: x_min,y_min,z_min,x_max,y_max,z_max')
    parser.add_argument('--num_points', type=int, default=1024,
                       help='Number of points after FPS downsampling')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA for FPS')

    args = parser.parse_args()

    # Parse scene bounds
    scene_bounds = [float(x) for x in args.scene_bounds.split(',')]
    assert len(scene_bounds) == 6, "Scene bounds must have 6 values"

    convert_franka_3cam_dataset(
        data_root=args.data_root,
        save_path=args.save_path,
        scene_bounds=scene_bounds,
        num_points=args.num_points,
        use_cuda=not args.no_cuda
    )
