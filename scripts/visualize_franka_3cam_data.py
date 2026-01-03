"""
Visualize Franka 3-camera dataset (both raw and processed zarr data)

This script supports visualization of any Franka task with 3 cameras (e.g., cook5, push_T_5, etc.)
"""
import os
import pickle
import zarr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from termcolor import cprint
import argparse


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


def visualize_raw_point_clouds(data_root, trail_name="trail_1", timestep=0, scene_bounds=None):
    """
    Visualize raw point clouds from 3 cameras before and after transformation
    """
    if scene_bounds is None:
        scene_bounds = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9]

    trail_path = Path(data_root) / trail_name

    # Load extrinsics
    with open(trail_path / "extrinsics.pkl", 'rb') as f:
        extrinsics = pickle.load(f)

    # Load point clouds from 3 cameras
    pcd_files = sorted(os.listdir(trail_path / "3rd_1_pcd"))
    pcd_file = pcd_files[timestep]

    fig = plt.figure(figsize=(20, 10))

    camera_names = ['3rd_1', '3rd_2', '3rd_3']
    colors = ['red', 'green', 'blue']

    # Plot original point clouds
    for idx, (cam_name, color) in enumerate(zip(camera_names, colors)):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

        # Load point cloud
        with open(trail_path / f"{cam_name}_pcd" / pcd_file, 'rb') as f:
            pcd = pickle.load(f)  # (H, W, 3)

        # Reshape and remove zeros
        pcd_flat = pcd.reshape(-1, 3)
        valid_mask = np.linalg.norm(pcd_flat, axis=1) > 0.01
        valid_pcd = pcd_flat[valid_mask]

        # Downsample for visualization
        if len(valid_pcd) > 5000:
            indices = np.random.choice(len(valid_pcd), 5000, replace=False)
            valid_pcd = valid_pcd[indices]

        ax.scatter(valid_pcd[:, 0], valid_pcd[:, 1], valid_pcd[:, 2],
                  c=color, s=1, alpha=0.5)
        ax.set_title(f'{cam_name} (camera frame)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Plot transformed and merged point clouds
    ax_merged = fig.add_subplot(2, 3, 4, projection='3d')
    ax_cropped = fig.add_subplot(2, 3, 5, projection='3d')

    all_transformed = []
    for cam_name, color in zip(camera_names, colors):
        # Load point cloud
        with open(trail_path / f"{cam_name}_pcd" / pcd_file, 'rb') as f:
            pcd = pickle.load(f)  # (H, W, 3)

        # Transform to robot base frame using extrinsics
        transformed_pcd = convert_pcd_to_base(extrinsics[cam_name], pcd)
        transformed = transformed_pcd.reshape(-1, 3)

        # Remove invalid
        valid_mask = np.linalg.norm(transformed, axis=1) > 0.01
        transformed_valid = transformed[valid_mask]

        # Downsample
        if len(transformed_valid) > 2000:
            indices = np.random.choice(len(transformed_valid), 2000, replace=False)
            transformed_valid = transformed_valid[indices]

        all_transformed.append(transformed_valid)
        ax_merged.scatter(transformed_valid[:, 0], transformed_valid[:, 1],
                         transformed_valid[:, 2], c=color, s=1, alpha=0.5)

    ax_merged.set_title('Merged (robot base frame)')
    ax_merged.set_xlabel('X')
    ax_merged.set_ylabel('Y')
    ax_merged.set_zlabel('Z')

    # Draw scene bounds box
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    from itertools import product
    for s, e in zip(product([x_min, x_max], [y_min, y_max], [z_min, z_max]),
                   product([x_min, x_max], [y_min, y_max], [z_min, z_max])):
        if sum([abs(s[i] - e[i]) for i in range(3)]) == abs(x_max - x_min):
            ax_merged.plot3D(*zip(s, e), color='black', linewidth=2)

    # Plot cropped point cloud
    merged = np.vstack(all_transformed)
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    mask = (
        (merged[:, 0] > x_min) & (merged[:, 0] < x_max) &
        (merged[:, 1] > y_min) & (merged[:, 1] < y_max) &
        (merged[:, 2] > z_min) & (merged[:, 2] < z_max)
    )
    cropped = merged[mask]

    if len(cropped) > 5000:
        indices = np.random.choice(len(cropped), 5000, replace=False)
        cropped = cropped[indices]

    ax_cropped.scatter(cropped[:, 0], cropped[:, 1], cropped[:, 2],
                      c='purple', s=1, alpha=0.5)
    ax_cropped.set_title(f'After cropping ({len(merged[mask])} points)')
    ax_cropped.set_xlabel('X')
    ax_cropped.set_ylabel('Y')
    ax_cropped.set_zlabel('Z')

    plt.suptitle(f'{trail_name} - Timestep {timestep}')
    plt.tight_layout()
    plt.savefig(f'franka_3cam_raw_visualization_{trail_name}_t{timestep}.png', dpi=150)
    cprint(f'Saved visualization to franka_3cam_raw_visualization_{trail_name}_t{timestep}.png', 'green')
    plt.show()


def visualize_zarr_data(zarr_path, episode_idx=0, timestep=0):
    """
    Visualize processed zarr data
    """
    cprint(f'Loading zarr data from {zarr_path}', 'cyan')
    zarr_root = zarr.open(zarr_path, 'r')

    point_clouds = zarr_root['data']['point_cloud'][:]
    states = zarr_root['data']['state'][:]
    actions = zarr_root['data']['action'][:]
    episode_ends = zarr_root['meta']['episode_ends'][:]

    cprint(f'Loaded {len(episode_ends)} episodes', 'green')
    cprint(f'Point cloud shape: {point_clouds.shape}', 'green')
    cprint(f'State shape: {states.shape}', 'green')
    cprint(f'Action shape: {actions.shape}', 'green')

    # Get episode range
    if episode_idx == 0:
        start_idx = 0
    else:
        start_idx = episode_ends[episode_idx - 1]
    end_idx = episode_ends[episode_idx]

    episode_length = end_idx - start_idx
    cprint(f'Episode {episode_idx}: {episode_length} timesteps', 'cyan')

    # Get data for this episode
    episode_pcd = point_clouds[start_idx:end_idx]
    episode_states = states[start_idx:end_idx]
    episode_actions = actions[start_idx:end_idx]

    # Create figure
    fig = plt.figure(figsize=(20, 8))

    # Plot point cloud at specific timestep
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    pcd = episode_pcd[timestep]
    valid_mask = np.linalg.norm(pcd, axis=1) > 0.01
    valid_pcd = pcd[valid_mask]

    ax1.scatter(valid_pcd[:, 0], valid_pcd[:, 1], valid_pcd[:, 2],
               c='blue', s=1, alpha=0.5)
    ax1.set_title(f'Point Cloud (t={timestep})\n{len(valid_pcd)} points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot robot trajectory (end-effector positions)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    positions = episode_states[:, :3]  # xyz
    ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Trajectory')
    ax2.scatter(positions[timestep, 0], positions[timestep, 1], positions[timestep, 2],
               c='red', s=100, marker='o', label=f't={timestep}')
    ax2.set_title('End-Effector Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    # Plot action and state distributions
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(episode_states[:, -1], label='Gripper state', linewidth=2)
    ax3.axvline(x=timestep, color='r', linestyle='--', label=f't={timestep}')
    ax3.set_title('Gripper State Over Time')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Gripper (0=closed, 1=open)')
    ax3.legend()
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 3, 6)
    action_norms = np.linalg.norm(episode_actions[:, :3], axis=1)  # position action norms
    ax4.plot(action_norms, label='Position action norm', linewidth=2)
    ax4.axvline(x=timestep, color='r', linestyle='--', label=f't={timestep}')
    ax4.set_title('Action Magnitude Over Time')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Action Norm')
    ax4.legend()
    ax4.grid(True)

    plt.suptitle(f'Episode {episode_idx} - Timestep {timestep}')
    plt.tight_layout()
    plt.savefig(f'franka_3cam_zarr_visualization_ep{episode_idx}_t{timestep}.png', dpi=150)
    cprint(f'Saved visualization to franka_3cam_zarr_visualization_ep{episode_idx}_t{timestep}.png', 'green')
    plt.show()


def print_zarr_statistics(zarr_path):
    """
    Print statistics of zarr dataset
    """
    zarr_root = zarr.open(zarr_path, 'r')

    point_clouds = zarr_root['data']['point_cloud'][:]
    states = zarr_root['data']['state'][:]
    actions = zarr_root['data']['action'][:]
    episode_ends = zarr_root['meta']['episode_ends'][:]

    cprint('\n' + '='*50, 'cyan')
    cprint('Dataset Statistics', 'cyan')
    cprint('='*50, 'cyan')

    cprint(f'\nNumber of episodes: {len(episode_ends)}', 'green')
    cprint(f'Total timesteps: {len(point_clouds)}', 'green')

    # Episode lengths
    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
    cprint(f'\nEpisode lengths:', 'green')
    cprint(f'  Min: {episode_lengths.min()}', 'white')
    cprint(f'  Max: {episode_lengths.max()}', 'white')
    cprint(f'  Mean: {episode_lengths.mean():.2f}', 'white')
    cprint(f'  Median: {np.median(episode_lengths):.2f}', 'white')

    # Point cloud statistics
    cprint(f'\nPoint cloud statistics:', 'green')
    cprint(f'  Shape: {point_clouds.shape}', 'white')
    cprint(f'  Min: {point_clouds.min():.3f}', 'white')
    cprint(f'  Max: {point_clouds.max():.3f}', 'white')
    cprint(f'  Mean: {point_clouds.mean():.3f}', 'white')
    cprint(f'  Std: {point_clouds.std():.3f}', 'white')

    # State statistics
    cprint(f'\nState statistics (8D: xyz + wxyz quaternion + gripper):', 'green')
    state_names = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper']
    for i, name in enumerate(state_names):
        cprint(f'  {name}: [{states[:, i].min():.3f}, {states[:, i].max():.3f}], mean={states[:, i].mean():.3f}', 'white')

    # Action statistics
    cprint(f'\nAction statistics (8D: delta xyz + delta quat + delta gripper):', 'green')
    action_names = ['dx', 'dy', 'dz', 'dqw', 'dqx', 'dqy', 'dqz', 'd_gripper']
    for i, name in enumerate(action_names):
        cprint(f'  {name}: [{actions[:, i].min():.3f}, {actions[:, i].max():.3f}], mean={actions[:, i].mean():.3f}', 'white')

    cprint('\n' + '='*50, 'cyan')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize Franka 3-camera dataset (raw or zarr format)'
    )
    parser.add_argument('--mode', type=str, choices=['raw', 'zarr', 'stats'], default='stats',
                       help='Visualization mode: raw (raw point clouds), zarr (processed data), stats (dataset statistics)')
    parser.add_argument('--data_root', type=str,
                       help='Path to raw dataset directory (e.g., /path/to/cook_5 or /path/to/push_T_5)')
    parser.add_argument('--zarr_path', type=str,
                       help='Path to zarr file (e.g., /path/to/cook5.zarr)')
    parser.add_argument('--trail', type=str, default='trail_1',
                       help='Trail name for raw visualization (e.g., trail_1)')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index for zarr visualization')
    parser.add_argument('--timestep', type=int, default=0,
                       help='Timestep for visualization')
    parser.add_argument('--scene_bounds', type=str,
                       default="-0.1,-0.5,-0.1,0.9,0.5,0.9",
                       help='Scene bounds for raw visualization: x_min,y_min,z_min,x_max,y_max,z_max')

    args = parser.parse_args()

    if args.mode == 'raw':
        if not args.data_root:
            parser.error("--data_root is required for 'raw' mode")
        scene_bounds = [float(x) for x in args.scene_bounds.split(',')]
        visualize_raw_point_clouds(args.data_root, args.trail, args.timestep, scene_bounds)
    elif args.mode == 'zarr':
        if not args.zarr_path:
            parser.error("--zarr_path is required for 'zarr' mode")
        visualize_zarr_data(args.zarr_path, args.episode, args.timestep)
    elif args.mode == 'stats':
        if not args.zarr_path:
            parser.error("--zarr_path is required for 'stats' mode")
        print_zarr_statistics(args.zarr_path)
