"""
Point Cloud Processor for Real Robot Deployment

This module handles real-time point cloud acquisition and preprocessing
from 3 ZED cameras, following the same pipeline as training data.
"""

import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
from typing import Tuple, List
from .camera_utils import Camera, get_cam_extrinsic


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


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """
    FPS sampling to downsample point cloud

    Args:
        points: (N, 3) numpy array
        num_points: target number of points
        use_cuda: whether to use CUDA acceleration
    Returns:
        sampled_points: (num_points, 3) numpy array
    """
    K = [num_points]
    if use_cuda and torch.cuda.is_available():
        points_tensor = torch.from_numpy(points).cuda().float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).cpu().numpy()
    else:
        points_tensor = torch.from_numpy(points).float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).numpy()

    return sampled_points.astype(np.float32)


class PointCloudProcessor:
    """
    Real-time point cloud processor for 3-camera setup

    Follows the same preprocessing pipeline as training:
    1. Acquire point clouds from 3 ZED cameras
    2. Transform to robot base frame using extrinsics
    3. Merge point clouds
    4. Remove invalid points
    5. Crop to scene bounds
    6. FPS downsample to target number of points
    """

    def __init__(
        self,
        scene_bounds: List[float] = None,
        num_points: int = 1024,
        use_cuda: bool = True,
        camera_ids: List[str] = None,
    ):
        """
        Initialize point cloud processor

        Args:
            scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
            num_points: target number of points after FPS downsampling
            use_cuda: whether to use CUDA for FPS
            camera_ids: list of camera IDs (default: ['3rd_1', '3rd_2', '3rd_3'])
        """
        if scene_bounds is None:
            # Default scene bounds (match training data)
            scene_bounds = [0.05, -0.5, 0.0, 0.85, 0.5, 0.8]

        if camera_ids is None:
            camera_ids = ['3rd_1', '3rd_2', '3rd_3']

        self.scene_bounds = scene_bounds
        self.num_points = num_points
        self.use_cuda = use_cuda
        self.camera_ids = camera_ids

        # Load camera extrinsics (4x4 transformation matrices)
        self.extrinsics = {}
        for cam_id in self.camera_ids:
            self.extrinsics[cam_id] = get_cam_extrinsic(cam_id)

        # Initialize camera
        self.camera = Camera()

        print(f"[PointCloudProcessor] Initialized:")
        print(f"  - Scene bounds: {scene_bounds}")
        print(f"  - Target points: {num_points}")
        print(f"  - Use CUDA: {use_cuda}")
        print(f"  - Camera IDs: {camera_ids}")

    def capture_point_clouds(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Capture point clouds and RGB images from all cameras

        Returns:
            pcd_list: List of point clouds [(H, W, 3), ...] for each camera
            rgb_list: List of RGB images [(H, W, 3), ...] for each camera
        """
        pcd_list = []
        rgb_list = []

        for cam_id in self.camera_ids:
            # Capture from camera
            data = self.camera.get_point_cloud_and_rgb(cam_id)

            if data is None or 'pcd' not in data or 'rgb' not in data:
                raise RuntimeError(f"Failed to capture from camera {cam_id}")

            pcd_list.append(data['pcd'])
            rgb_list.append(data['rgb'])

        return pcd_list, rgb_list

    def process_point_clouds(
        self,
        pcd_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Process point clouds from 3 cameras following training pipeline

        Args:
            pcd_list: List of point clouds [(H, W, 3), ...] from 3 cameras

        Returns:
            final_pcd: (num_points, 3) processed point cloud in robot base frame
        """
        merged_points = []

        for cam_id, pcd in zip(self.camera_ids, pcd_list):
            # Transform to robot base frame
            transformed = convert_pcd_to_base(self.extrinsics[cam_id], pcd)

            # Reshape to (N, 3)
            transformed_flat = transformed.reshape(-1, 3)

            # Remove invalid points (close to origin)
            valid_mask = np.linalg.norm(transformed_flat, axis=1) > 0.01
            valid_points = transformed_flat[valid_mask]

            merged_points.append(valid_points)

        # Merge all cameras
        merged_points = np.vstack(merged_points)

        # Crop to scene bounds
        cropped_points = crop_point_cloud(merged_points, self.scene_bounds)

        # Check if we have enough points
        if len(cropped_points) < self.num_points:
            print(f"[PointCloudProcessor] Warning: Only {len(cropped_points)} points "
                  f"after cropping, padding to {self.num_points}")
            # Pad with zeros if not enough points
            pad_size = self.num_points - len(cropped_points)
            padding = np.zeros((pad_size, 3), dtype=np.float32)
            final_pcd = np.vstack([cropped_points, padding])
            return final_pcd.astype(np.float32)

        # FPS downsampling
        final_pcd = farthest_point_sampling(
            cropped_points, self.num_points, self.use_cuda
        )

        return final_pcd

    def get_observation(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Get current observation (point cloud + RGB images)

        Returns:
            point_cloud: (num_points, 3) processed point cloud
            rgb_images: List of RGB images for visualization/debugging
        """
        # Capture from cameras
        pcd_list, rgb_list = self.capture_point_clouds()

        # Process point clouds
        point_cloud = self.process_point_clouds(pcd_list)

        return point_cloud, rgb_list

    def close(self):
        """Close camera connections"""
        if self.camera is not None:
            self.camera.stop()
            print("[PointCloudProcessor] Camera closed")
