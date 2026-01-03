# Franka 3-Camera Dataset Processing

This directory contains generic scripts for processing Franka robot datasets with 3-camera point cloud observations.

These scripts support **any Franka task** following the standard format (e.g., `cook_5`, `push_T_5`, etc.)

## Data Format Requirements

Your dataset should follow this structure:

```
<task_name>/
├── trail_1/
│   ├── extrinsics.pkl           # Camera extrinsics (4x4 matrices for 3rd_1, 3rd_2, 3rd_3)
│   ├── poses/                   # End-effector poses [x, y, z, qw, qx, qy, qz]
│   ├── gripper_states/          # Gripper states (boolean)
│   ├── 3rd_1_pcd/              # Point clouds from camera 1
│   ├── 3rd_2_pcd/              # Point clouds from camera 2
│   ├── 3rd_3_pcd/              # Point clouds from camera 3
│   ├── 3rd_1_bgr/              # (Optional) RGB images from camera 1
│   ├── 3rd_2_bgr/              # (Optional) RGB images from camera 2
│   └── 3rd_3_bgr/              # (Optional) RGB images from camera 3
├── trail_2/
│   └── ...
└── ...
```

## Scripts

### 1. Check Dataset Format

Analyze the structure and format of your dataset:

```bash
python check_franka_3cam_format.py \
    --data_root /path/to/your/task_name
```

**Example for cook_5:**
```bash
python check_franka_3cam_format.py \
    --data_root /DATA/disk0/lpy/data/Franka_data_3zed_5/cook_5
```

**Example for push_T_5:**
```bash
python check_franka_3cam_format.py \
    --data_root /DATA/disk0/lpy/data/Franka_data_3zed_5/push_T_5
```

### 2. Convert Dataset to Zarr Format

Convert raw dataset to zarr format for 3D-Diffusion-Policy training:

```bash
python convert_franka_3cam_data.py \
    --data_root /path/to/raw/task_name \
    --save_path /path/to/output/task_name.zarr \
    --scene_bounds "x_min,y_min,z_min,x_max,y_max,z_max" \
    --num_points 1024
```

**Example for cook_5:**
```bash
python convert_franka_3cam_data.py \
    --data_root /DATA/disk0/lpy/data/Franka_data_3zed_5/cook_5 \
    --save_path /DATA/disk0/lpy/baseline/3D-Diffusion-Policy/3D-Diffusion-Policy/data/cook5.zarr \
    --scene_bounds "-0.1,-0.5,-0.1,0.9,0.5,0.9" \
    --num_points 1024
```

**Example for push_T_5:**
```bash
python convert_franka_3cam_data.py \
    --data_root /DATA/disk0/lpy/data/Franka_data_3zed_5/push_T_5 \
    --save_path /DATA/disk0/lpy/baseline/3D-Diffusion-Policy/3D-Diffusion-Policy/data/push_t5.zarr \
    --scene_bounds "-0.1,-0.5,-0.1,0.9,0.5,0.9" \
    --num_points 1024
```

### 3. Visualize Dataset

Visualize raw point clouds, processed zarr data, or dataset statistics:

**Check statistics:**
```bash
python visualize_franka_3cam_data.py \
    --mode stats \
    --zarr_path /path/to/task_name.zarr
```

**Visualize processed zarr data:**
```bash
python visualize_franka_3cam_data.py \
    --mode zarr \
    --zarr_path /path/to/task_name.zarr \
    --episode 0 \
    --timestep 10
```

**Visualize raw point clouds:**
```bash
python visualize_franka_3cam_data.py \
    --mode raw \
    --data_root /path/to/raw/task_name \
    --trail trail_1 \
    --timestep 0 \
    --scene_bounds "-0.1,-0.5,-0.1,0.9,0.5,0.9"
```

## Creating Task-Specific Configuration

After converting your dataset, create a task-specific YAML config:

1. Copy the template: `diffusion_policy_3d/config/task/franka_cook5.yaml`
2. Rename to your task: `franka_<your_task>.yaml`
3. Update the `zarr_path` to point to your converted data
4. Adjust `task_name` and other task-specific parameters

Example for `push_T_5`:

```yaml
name: franka_push_t5
task_name: push_t5

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [1024, 3]
      type: point_cloud
    agent_pos:
      shape: [8]
      type: low_dim
  action:
    shape: [8]

env_runner: null

dataset:
  _target_: diffusion_policy_3d.dataset.franka_3cam_dataset.Franka3CamDataset
  zarr_path: /DATA/disk0/lpy/baseline/3D-Diffusion-Policy/3D-Diffusion-Policy/data/push_t5.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.1
  max_train_episodes: null
```

## Processing Pipeline

The conversion script performs the following steps:

1. **Load point clouds** from 3 cameras (3rd_1, 3rd_2, 3rd_3)
2. **Transform** each point cloud to robot base frame using extrinsics
3. **Merge** point clouds from all cameras
4. **Remove invalid points** (close to origin)
5. **Crop** to scene bounds (workspace)
6. **FPS downsample** to target number of points (default: 1024)
7. **Compute actions** as relative deltas between consecutive poses
8. **Save to zarr** format with compression

## Notes

- The `Franka3CamDataset` class in `diffusion_policy_3d/dataset/franka_3cam_dataset.py` is generic and works for all tasks
- Scene bounds should be adjusted based on your robot's workspace
- The number of FPS points (default 1024) can be adjusted based on your compute budget
- All scripts use the same `convert_pcd_to_base()` function from BridgeVLA codebase for consistency
