#!/bin/bash

# Real Robot Evaluation Script for 3D-Diffusion-Policy
# Usage: bash scripts/run_real_robot.sh [config_file] [checkpoint_path] [device]

# Default parameters
CONFIG_FILE="${1:-scripts/real_robot_config.yaml}"
CHECKPOINT="${2:-}"
DEVICE="${3:-cuda:0}"

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# ZED SDK and PyTorch compatibility settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Navigate to project root
cd "$(dirname "$0")/.."

echo "========================================"
echo "3D-Diffusion-Policy Real Robot Evaluation"
echo "========================================"
echo "Config file: $CONFIG_FILE"
echo "Device: $DEVICE"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "========================================"
echo ""

# Build command
CMD="python scripts/real_robot_eval.py --config $CONFIG_FILE --device $DEVICE"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Run evaluation
echo "Launching evaluation..."
echo "Command: $CMD"
echo ""

$CMD
