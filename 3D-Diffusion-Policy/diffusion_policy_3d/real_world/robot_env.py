"""
Robot Environment for Real Robot Deployment

This module provides a unified interface for interacting with the real robot,
integrating point cloud acquisition, robot control, and safety monitoring.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple, Optional
import time

from .point_cloud_processor import PointCloudProcessor
from .robot_interface import RobotController, action_to_pose


class RobotEnv:
    """
    Robot environment for real-world deployment

    Integrates:
    - Point cloud processor (3 ZED cameras)
    - Robot controller (Franka robot)
    - Safety monitoring
    """

    def __init__(
        self,
        scene_bounds: list = None,
        num_points: int = 1024,
        use_cuda: bool = True,
        safety_bounds: list = None,
        control_hz: float = 10.0,
        action_scale: float = 1.0,
    ):
        """
        Initialize robot environment

        Args:
            scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] for point cloud cropping
            num_points: number of points after FPS
            use_cuda: whether to use CUDA for FPS
            safety_bounds: safety workspace bounds for robot
            control_hz: control loop frequency (Hz)
            action_scale: scaling factor for actions
        """
        if scene_bounds is None:
            scene_bounds = [0.05, -0.5, 0.0, 0.85, 0.5, 0.8]

        if safety_bounds is None:
            # Slightly larger than scene_bounds for safety
            safety_bounds = [0.0, -0.6, -0.05, 0.9, 0.6, 0.85]

        self.scene_bounds = scene_bounds
        self.safety_bounds = safety_bounds
        self.control_hz = control_hz
        self.control_dt = 1.0 / control_hz
        self.action_scale = action_scale

        # Initialize point cloud processor
        print("[RobotEnv] Initializing point cloud processor...")
        self.pcd_processor = PointCloudProcessor(
            scene_bounds=scene_bounds,
            num_points=num_points,
            use_cuda=use_cuda
        )

        # Initialize robot controller
        print("[RobotEnv] Initializing robot controller...")
        self.robot = RobotController()

        print(f"[RobotEnv] Initialized:")
        print(f"  - Scene bounds: {scene_bounds}")
        print(f"  - Safety bounds: {safety_bounds}")
        print(f"  - Control frequency: {control_hz} Hz")
        print(f"  - Action scale: {action_scale}")

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation

        Returns:
            obs_dict containing:
                - point_cloud: (1024, 3) point cloud in robot base frame
                - agent_pos: (8,) robot state [x, y, z, qw, qx, qy, qz, gripper]
        """
        # Get point cloud
        point_cloud, rgb_images = self.pcd_processor.get_observation()

        # Get robot state
        agent_pos = self.get_robot_state()

        obs = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
            'rgb_images': rgb_images  # For visualization/debugging
        }

        return obs

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state

        Returns:
            state: (8,) [x, y, z, qw, qx, qy, qz, gripper]
        """
        # Get current pose from robot
        current_pose = self.robot.get_current_pose()

        # current_pose format: [x, y, z, qx, qy, qz, qw, gripper]
        # Need to convert to: [x, y, z, qw, qx, qy, qz, gripper]

        x, y, z = current_pose[:3]
        qx, qy, qz, qw = current_pose[3:7]
        gripper = current_pose[7]

        state = np.array([x, y, z, qw, qx, qy, qz, gripper], dtype=np.float32)

        return state

    def check_safety(self, pose: np.ndarray) -> bool:
        """
        Check if pose is within safety bounds

        Args:
            pose: (7,) or (8,) pose [x, y, z, qw, qx, qy, qz, (gripper)]

        Returns:
            safe: True if within bounds
        """
        x, y, z = pose[:3]
        x_min, y_min, z_min, x_max, y_max, z_max = self.safety_bounds

        safe = (
            x_min < x < x_max and
            y_min < y < y_max and
            z_min < z < z_max
        )

        if not safe:
            print(f"[RobotEnv] Safety violation! Pose: [{x:.3f}, {y:.3f}, {z:.3f}]")
            print(f"[RobotEnv] Safety bounds: {self.safety_bounds}")

        return safe

    def step(
        self,
        action: np.ndarray,
        check_safety: bool = True
    ) -> Dict:
        """
        Execute one action step

        Args:
            action: (8,) delta action [dx, dy, dz, dqw, dqx, dqy, dqz, d_gripper]
            check_safety: whether to perform safety check

        Returns:
            step_info: dict with execution status
        """
        # Scale action
        scaled_action = action * self.action_scale

        # Get current robot state
        current_state = self.get_robot_state()

        # Compute target pose (current + delta)
        target_pose = self.compute_target_pose(current_state, scaled_action)

        # Safety check
        if check_safety and not self.check_safety(target_pose):
            print("[RobotEnv] Action rejected due to safety check!")
            return {'success': False, 'reason': 'safety_violation'}

        # Execute action on robot
        # Convert target_pose to robot format: [x, y, z, qx, qy, qz, qw, gripper]
        x, y, z, qw, qx, qy, qz, gripper = target_pose
        robot_pose = np.array([x, y, z, qx, qy, qz, qw, gripper], dtype=np.float32)

        try:
            self.robot.move_to_pose(robot_pose)
            success = True
        except Exception as e:
            print(f"[RobotEnv] Robot execution error: {e}")
            success = False

        # Wait for control dt
        time.sleep(self.control_dt)

        return {'success': success}

    def compute_target_pose(
        self,
        current_state: np.ndarray,
        delta_action: np.ndarray
    ) -> np.ndarray:
        """
        Compute target pose from current state and delta action

        Args:
            current_state: (8,) [x, y, z, qw, qx, qy, qz, gripper]
            delta_action: (8,) [dx, dy, dz, dqw, dqx, dqy, dqz, d_gripper]

        Returns:
            target_pose: (8,) [x, y, z, qw, qx, qy, qz, gripper]
        """
        # Position: simple addition
        target_pos = current_state[:3] + delta_action[:3]

        # Rotation: quaternion addition (simple approach, can be improved)
        # For small delta, this approximation is acceptable
        current_quat = current_state[3:7]  # [qw, qx, qy, qz]
        delta_quat = delta_action[3:7]     # [dqw, dqx, dqy, dqz]
        target_quat = current_quat + delta_quat

        # Normalize quaternion
        target_quat = target_quat / np.linalg.norm(target_quat)

        # Gripper: simple addition and clip
        target_gripper = np.clip(current_state[7] + delta_action[7], 0.0, 1.0)

        target_pose = np.concatenate([target_pos, target_quat, [target_gripper]])

        return target_pose

    def reset(self):
        """Reset environment (move to home pose, etc.)"""
        print("[RobotEnv] Resetting environment...")
        # Could add home pose reset here if needed
        pass

    def close(self):
        """Close all connections"""
        print("[RobotEnv] Closing environment...")
        self.pcd_processor.close()
        self.robot.close()
        print("[RobotEnv] Environment closed")
