from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset


class Franka3CamDataset(BaseDataset):
    """
    Generic dataset for Franka robot tasks with 3-camera point cloud observations

    This dataset class supports any Franka task with the same data format:
    - 3 cameras providing point clouds fused into robot base frame
    - End-effector pose (xyz + quaternion) + gripper state
    - Delta actions (position, rotation, gripper)

    Data format:
        - point_cloud: (N, 1024, 3) - xyz coordinates in robot base frame
        - state: (N, 8) - [x, y, z, qw, qx, qy, qz, gripper]
        - action: (N, 8) - [dx, dy, dz, dqw, dqx, dqy, dqz, d_gripper]

    Examples: cook5, push_T_5, or any other task following this format
    """
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name

        # Load replay buffer from zarr
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud'])

        # Create validation mask
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # Downsample training episodes if needed
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Create sequence sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """Create validation dataset"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Get normalizer for the dataset
        Normalizes action, agent_pos (state), and point_cloud
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],  # Use 'agent_pos' to match DP3 convention
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Optional: You can set point_cloud normalizer to identity
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """Convert replay buffer sample to training data format"""
        # State: (T, 8) - [x, y, z, qw, qx, qy, qz, gripper]
        agent_pos = sample['state'].astype(np.float32)

        # Point cloud: (T, 1024, 3) - [x, y, z]
        point_cloud = sample['point_cloud'].astype(np.float32)

        # Action: (T, 8) - [dx, dy, dz, dqw, dqx, dqy, dqz, d_gripper]
        action = sample['action'].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,  # T, 1024, 3
                'agent_pos': agent_pos,      # T, 8
            },
            'action': action                 # T, 8
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
