"""
DP3 Policy for Real Robot Deployment

This module loads trained DP3 model and performs action prediction in real-time.
"""

import torch
import dill
import numpy as np
from pathlib import Path
from typing import Dict
from collections import deque


class DP3RealWorldPolicy:
    """
    DP3 Policy wrapper for real robot deployment

    Handles:
    - Loading checkpoint and normalizer
    - Managing observation history buffer
    - Predicting actions from observations
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
    ):
        """
        Initialize DP3 policy for real robot

        Args:
            checkpoint_path: path to checkpoint file (e.g., 'data/outputs/.../checkpoints/latest.ckpt')
            device: device to run model on
            n_obs_steps: number of observation steps (history frames)
            n_action_steps: number of action steps to execute
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Load checkpoint
        print(f"[DP3Policy] Loading checkpoint from {checkpoint_path}")
        self.load_checkpoint()

        # Initialize observation history buffer
        self.obs_buffer = deque(maxlen=n_obs_steps)

        print(f"[DP3Policy] Initialized:")
        print(f"  - Device: {device}")
        print(f"  - n_obs_steps: {n_obs_steps}")
        print(f"  - n_action_steps: {n_action_steps}")
        print(f"  - Model loaded successfully")

    def load_checkpoint(self):
        """Load model from checkpoint"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Load checkpoint
        payload = torch.load(
            self.checkpoint_path.open('rb'),
            map_location=self.device,
            pickle_module=dill
        )

        # Extract model and normalizer
        cfg = payload['cfg']
        state_dicts = payload['state_dicts']

        # Import DP3 policy
        from diffusion_policy_3d.policy.dp3 import DP3

        # Create model
        self.model = DP3(cfg.policy)

        # Load state dict
        if 'model' in state_dicts:
            self.model.load_state_dict(state_dicts['model'])
        elif 'ema_model' in state_dicts:
            # Use EMA model if available
            self.model.load_state_dict(state_dicts['ema_model'])
            print("[DP3Policy] Using EMA model")
        else:
            raise ValueError("No model found in checkpoint")

        # Load normalizer
        if 'normalizer' in state_dicts:
            self.model.normalizer.load_state_dict(state_dicts['normalizer'])
        else:
            print("[DP3Policy] Warning: No normalizer found in checkpoint")

        # Set model to eval mode
        self.model.eval()
        self.model.to(self.device)

        # Store config for reference
        self.cfg = cfg

    def reset(self):
        """Reset observation buffer"""
        self.obs_buffer.clear()
        print("[DP3Policy] Observation buffer reset")

    def add_observation(self, point_cloud: np.ndarray, agent_pos: np.ndarray):
        """
        Add new observation to history buffer

        Args:
            point_cloud: (1024, 3) point cloud
            agent_pos: (8,) robot state [x, y, z, qw, qx, qy, qz, gripper]
        """
        obs = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos
        }
        self.obs_buffer.append(obs)

    def is_ready(self) -> bool:
        """Check if enough observations are collected"""
        return len(self.obs_buffer) >= self.n_obs_steps

    def predict_action(self) -> np.ndarray:
        """
        Predict actions from current observation buffer

        Returns:
            actions: (n_action_steps, 8) array of delta actions
                     [dx, dy, dz, dqw, dqx, dqy, dqz, d_gripper]
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Not enough observations. Have {len(self.obs_buffer)}, "
                f"need {self.n_obs_steps}"
            )

        # Stack observations from buffer
        obs_list = list(self.obs_buffer)

        # Stack point clouds: (n_obs_steps, 1024, 3)
        point_clouds = np.stack([obs['point_cloud'] for obs in obs_list], axis=0)

        # Stack agent positions: (n_obs_steps, 8)
        agent_positions = np.stack([obs['agent_pos'] for obs in obs_list], axis=0)

        # Convert to torch tensors and add batch dimension
        obs_dict = {
            'point_cloud': torch.from_numpy(point_clouds).float().unsqueeze(0).to(self.device),
            'agent_pos': torch.from_numpy(agent_positions).float().unsqueeze(0).to(self.device)
        }

        # Predict actions
        with torch.no_grad():
            result = self.model.predict_action(obs_dict)
            action = result['action']  # (1, n_action_steps, 8)

        # Convert to numpy and remove batch dimension
        action = action.cpu().numpy()[0]  # (n_action_steps, 8)

        return action

    def get_action(
        self,
        point_cloud: np.ndarray,
        agent_pos: np.ndarray
    ) -> np.ndarray:
        """
        Convenience method: add observation and predict action

        Args:
            point_cloud: (1024, 3) point cloud
            agent_pos: (8,) robot state

        Returns:
            actions: (n_action_steps, 8) delta actions
        """
        # Add observation
        self.add_observation(point_cloud, agent_pos)

        # Predict if ready
        if self.is_ready():
            return self.predict_action()
        else:
            # Not ready yet, return zero actions
            print(f"[DP3Policy] Warming up... {len(self.obs_buffer)}/{self.n_obs_steps}")
            return np.zeros((self.n_action_steps, 8), dtype=np.float32)
