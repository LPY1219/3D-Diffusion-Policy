# Real Robot Deployment Guide for 3D-Diffusion-Policy

This guide explains how to deploy trained DP3 models on a real Franka robot with 3 ZED cameras.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Real Robot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐    ┌────────────┐  │
│  │ ZED Camera 1 │      │ ZED Camera 2 │    │ ZED Cam 3  │  │
│  └──────┬───────┘      └──────┬───────┘    └─────┬──────┘  │
│         │                     │                    │         │
│         └─────────────────────┼────────────────────┘         │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │ PointCloudProcessor │                   │
│                    │  - Transform to base│                   │
│                    │  - Merge 3 cameras  │                   │
│                    │  - Crop & FPS       │                   │
│                    └──────────┬──────────┘                   │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │   DP3 Policy        │                   │
│                    │  - Load checkpoint  │                   │
│                    │  - Predict actions  │                   │
│                    └──────────┬──────────┘                   │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │   RobotEnv          │                   │
│                    │  - Safety check     │                   │
│                    │  - Execute actions  │                   │
│                    └──────────┬──────────┘                   │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │   Franka Robot      │                   │
│                    └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Files Structure

```
3D-Diffusion-Policy/
├── scripts/
│   ├── real_robot_eval.py          # Main deployment script
│   ├── real_robot_config.yaml      # Configuration file
│   ├── run_real_robot.sh           # Launch script
│   └── README_real_robot.md        # This file
└── 3D-Diffusion-Policy/
    └── diffusion_policy_3d/
        └── real_world/
            ├── __init__.py
            ├── camera_utils.py             # Camera interface (copied from BridgeVLA)
            ├── robot_interface.py          # Robot controller (copied from BridgeVLA)
            ├── point_cloud_processor.py    # Point cloud acquisition & preprocessing
            ├── dp3_policy.py               # Model loading & inference
            └── robot_env.py                # Unified robot environment
```

## Quick Start

### 1. Prepare Environment

```bash
# Activate conda environment
conda activate dp3_lpy

# Navigate to project root
cd /DATA/disk0/lpy/baseline/3D-Diffusion-Policy
```

### 2. Check Configuration

Edit `scripts/real_robot_config.yaml` to match your setup:

```yaml
model:
  checkpoint_path: "path/to/your/checkpoint.ckpt"
  device: "cuda:0"
  n_obs_steps: 2
  n_action_steps: 8

environment:
  scene_bounds: [0.05, -0.5, 0.0, 0.85, 0.5, 0.8]  # Must match training
  num_points: 1024                                   # Must match training
  control_hz: 10.0
  action_scale: 1.0  # Reduce if actions too aggressive

evaluation:
  max_steps: 100
  execution_steps: 8
  save_trajectory: true
```

### 3. Run Deployment

**Option 1: Use launch script (recommended)**

```bash
# Use default config
bash scripts/run_real_robot.sh

# Specify config file
bash scripts/run_real_robot.sh scripts/real_robot_config.yaml

# Specify checkpoint and device
bash scripts/run_real_robot.sh scripts/real_robot_config.yaml \
    "3D-Diffusion-Policy/data/outputs/franka_cook5-dp3-cook_5_seed0/checkpoints/latest.ckpt" \
    "cuda:0"
```

**Option 2: Run Python script directly**

```bash
python scripts/real_robot_eval.py \
    --config scripts/real_robot_config.yaml \
    --checkpoint "3D-Diffusion-Policy/data/outputs/franka_cook5-dp3-cook_5_seed0/checkpoints/latest.ckpt" \
    --device cuda:0
```

## How It Works

### Control Loop

```python
1. Initialize environment and policy
2. Warmup: Collect n_obs_steps observations
3. Main loop (max_steps iterations):
   a. Get observation (point cloud + robot state)
   b. Predict n_action_steps actions
   c. Execute first execution_steps actions
   d. Log and monitor
4. Save trajectory and cleanup
```

### Data Flow

```
Camera Capture → Point Cloud Processing → Model Inference → Robot Execution
     (3 cams)        (1024 pts, base frame)   (delta actions)     (move to pose)
```

### Key Features

- **Real-time processing**: 10Hz control loop
- **Safety checks**: Workspace bounds validation
- **History buffer**: Maintains n_obs_steps observations
- **Action chunking**: Predicts N steps, executes M steps (M ≤ N)
- **Trajectory logging**: Saves data for offline analysis

## Data Format Alignment

**Critical**: Deployment must match training data preprocessing exactly!

| Item | Training | Deployment | Status |
|------|----------|------------|--------|
| Point cloud shape | (1024, 3) | (1024, 3) | ✅ |
| Scene bounds | [0.05,-0.5,0,0.85,0.5,0.8] | Same | ✅ |
| Coordinate frame | Robot base | Robot base | ✅ |
| State format | [xyz, qw qx qy qz, gripper] | Same | ✅ |
| Action format | Delta [dxyz, dquat, dgripper] | Same | ✅ |
| Quaternion order | wxyz | wxyz | ✅ |

## Troubleshooting

### Camera Issues

**Problem**: Failed to capture from camera

**Solution**:
1. Check camera connections (USB 3.0)
2. Verify ZED SDK installation: `python -c "import pyzed.sl as sl"`
3. Check camera serial numbers in `camera_utils.py`

### Model Issues

**Problem**: Checkpoint not found

**Solution**:
1. Verify checkpoint path in config file
2. Check training outputs: `ls 3D-Diffusion-Policy/data/outputs/*/checkpoints/`

**Problem**: CUDA out of memory

**Solution**:
1. Use smaller batch size (already 1 for deployment)
2. Try CPU: `--device cpu` (slower but works)

### Robot Issues

**Problem**: Robot not responding

**Solution**:
1. Check robot connection and FCI setup
2. Verify robot_interface.py configuration
3. Test with simple robot movements first

**Problem**: Safety violation errors

**Solution**:
1. Review `safety_bounds` in config
2. Check current robot position
3. Adjust `scene_bounds` if needed

### Action Issues

**Problem**: Actions too aggressive/weak

**Solution**:
1. Adjust `action_scale` in config (try 0.5 or 1.5)
2. Check `execution_steps` (reduce if too fast)
3. Verify training data action distribution

## Advanced Usage

### Custom Task

To deploy on a different task (e.g., push_T_5):

1. Train model on new task data
2. Update checkpoint path in config
3. Adjust scene_bounds if workspace different
4. Test with reduced action_scale first

### Visualization

Enable visualization in config:

```yaml
evaluation:
  visualize: true
```

This will display camera feeds and point clouds during execution.

### Trajectory Analysis

Trajectories are saved to `data/real_robot_trajectories/`:

```python
import numpy as np

# Load trajectory
data = np.load("data/real_robot_trajectories/trajectory_20250104_123456.npz", allow_pickle=True)
trajectory = data['trajectory']
config = data['config'].item()

# Analyze
for step in trajectory:
    print(f"Step {step['step']}: pos={step['observation']['agent_pos'][:3]}")
```

## Safety Notes

⚠️ **IMPORTANT SAFETY GUIDELINES**:

1. **Always have emergency stop ready**
2. **Start with low action_scale (e.g., 0.5)**
3. **Test in safe environment first**
4. **Keep workspace clear of obstacles**
5. **Monitor robot during execution**
6. **Verify safety_bounds before running**

## Performance Tips

1. **Use CUDA**: Ensures fast point cloud processing
2. **Optimize control_hz**: 10Hz is good balance
3. **Tune execution_steps**: More = smoother but slower reaction
4. **Monitor inference time**: Should be < 100ms

## Contact & Support

For issues or questions about real robot deployment:
- Check training logs to verify model quality
- Review configuration alignment with training
- Test individual components (camera, robot, model) separately

## Example Session

```bash
$ bash scripts/run_real_robot.sh
========================================
3D-Diffusion-Policy Real Robot Evaluation
========================================
Config file: scripts/real_robot_config.yaml
Device: cuda:0
========================================

Launching evaluation...

[DP3Policy] Loading checkpoint from 3D-Diffusion-Policy/data/outputs/franka_cook5-dp3-cook_5_seed0/checkpoints/latest.ckpt
[DP3Policy] Initialized successfully

[PointCloudProcessor] Initialized with 3 cameras
[RobotEnv] Initialized

[Evaluator] Warming up policy...
[Evaluator] Warmup 1/2
[Evaluator] Warmup 2/2
[Evaluator] Warmup complete!

[Evaluator] === Step 1/100 ===
[Evaluator] Robot state: [0.450, 0.120, 0.350]
[Evaluator] Inference time: 45.2 ms
[Evaluator] Predicted 8 actions
[Evaluator] Executing first 8 actions...
  Action 1/8: pos_delta=[0.002, -0.001, 0.003], gripper_delta=0.000
  ...

[Evaluator] Evaluation complete!
[Evaluator] Trajectory saved to data/real_robot_trajectories/trajectory_20250104_123456.npz
```
