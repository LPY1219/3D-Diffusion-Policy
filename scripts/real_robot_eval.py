"""
Real Robot Evaluation Script for 3D-Diffusion-Policy

This script deploys the trained DP3 model on a real Franka robot with 3 ZED cameras.

Usage:
    python scripts/real_robot_eval.py --checkpoint <path_to_checkpoint> --config <config_file>
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import yaml
import numpy as np
import time
from datetime import datetime
from termcolor import cprint

# Import real-world modules
sys.path.insert(0, os.path.join(ROOT_DIR, "3D-Diffusion-Policy"))
from diffusion_policy_3d.real_world.dp3_policy import DP3RealWorldPolicy
from diffusion_policy_3d.real_world.robot_env import RobotEnv


class RealRobotEvaluator:
    """
    Real robot evaluator for DP3 policy

    Manages the full evaluation loop:
    1. Initialize environment and policy
    2. Run control loop
    3. Monitor and log results
    """

    def __init__(self, config: dict):
        """
        Initialize evaluator

        Args:
            config: configuration dictionary
        """
        self.config = config

        # Extract config
        model_config = config['model']
        env_config = config['environment']
        eval_config = config['evaluation']

        # Initialize policy
        cprint("[Evaluator] Initializing DP3 policy...", "cyan")
        self.policy = DP3RealWorldPolicy(
            checkpoint_path=model_config['checkpoint_path'],
            device=model_config['device'],
            n_obs_steps=model_config['n_obs_steps'],
            n_action_steps=model_config['n_action_steps'],
        )

        # Initialize environment
        cprint("[Evaluator] Initializing robot environment...", "cyan")
        self.env = RobotEnv(
            scene_bounds=env_config['scene_bounds'],
            num_points=env_config['num_points'],
            use_cuda=env_config['use_cuda'],
            safety_bounds=env_config['safety_bounds'],
            control_hz=env_config['control_hz'],
            action_scale=env_config['action_scale'],
        )

        # Evaluation parameters
        self.max_steps = eval_config['max_steps']
        self.execution_steps = eval_config['execution_steps']
        self.visualize = eval_config.get('visualize', False)
        self.save_trajectory = eval_config.get('save_trajectory', False)

        # Logging
        self.trajectory_log = []

        cprint("[Evaluator] Initialization complete!", "green")

    def warmup(self):
        """
        Warm up the policy by collecting initial observations
        """
        cprint(f"[Evaluator] Warming up policy (need {self.policy.n_obs_steps} observations)...", "yellow")

        while not self.policy.is_ready():
            # Get observation
            obs = self.env.get_observation()

            # Add to policy buffer
            self.policy.add_observation(
                obs['point_cloud'],
                obs['agent_pos']
            )

            cprint(f"[Evaluator] Warmup {len(self.policy.obs_buffer)}/{self.policy.n_obs_steps}", "yellow")

            time.sleep(0.1)

        cprint("[Evaluator] Warmup complete!", "green")

    def run_evaluation(self):
        """
        Run full evaluation episode
        """
        cprint("="*60, "cyan")
        cprint("Starting Real Robot Evaluation", "cyan")
        cprint("="*60, "cyan")

        # Reset
        self.policy.reset()
        self.env.reset()

        # Warmup
        self.warmup()

        # Main control loop
        step_count = 0
        total_actions_executed = 0

        try:
            while step_count < self.max_steps:
                cprint(f"\n[Evaluator] === Step {step_count+1}/{self.max_steps} ===", "cyan")

                # Get current observation
                obs = self.env.get_observation()

                # Log current state
                cprint(f"[Evaluator] Robot state: {obs['agent_pos'][:3]}", "white")

                # Predict actions
                t_start = time.time()
                actions = self.policy.get_action(
                    obs['point_cloud'],
                    obs['agent_pos']
                )
                t_inference = time.time() - t_start

                cprint(f"[Evaluator] Inference time: {t_inference*1000:.1f} ms", "yellow")
                cprint(f"[Evaluator] Predicted {len(actions)} actions", "white")

                # Execute actions (first N steps)
                num_exec = min(self.execution_steps, len(actions))
                cprint(f"[Evaluator] Executing first {num_exec} actions...", "green")

                for i, action in enumerate(actions[:num_exec]):
                    cprint(f"  Action {i+1}/{num_exec}: pos_delta={action[:3]}, gripper_delta={action[7]:.3f}", "white")

                    # Execute action
                    step_info = self.env.step(action, check_safety=True)

                    if not step_info['success']:
                        cprint(f"[Evaluator] Action execution failed!", "red")
                        if 'reason' in step_info:
                            cprint(f"[Evaluator] Reason: {step_info['reason']}", "red")
                        break

                    total_actions_executed += 1

                # Log trajectory
                if self.save_trajectory:
                    self.trajectory_log.append({
                        'step': step_count,
                        'observation': obs,
                        'actions': actions,
                        'num_executed': num_exec,
                        'timestamp': time.time()
                    })

                step_count += 1

                # Check termination condition (can be customized)
                # For now, just run for max_steps

        except KeyboardInterrupt:
            cprint("\n[Evaluator] Evaluation interrupted by user", "red")

        except Exception as e:
            cprint(f"\n[Evaluator] Error during evaluation: {e}", "red")
            import traceback
            traceback.print_exc()

        finally:
            # Summary
            cprint("\n" + "="*60, "cyan")
            cprint("Evaluation Summary", "cyan")
            cprint("="*60, "cyan")
            cprint(f"Total steps: {step_count}", "white")
            cprint(f"Total actions executed: {total_actions_executed}", "white")

            # Save trajectory
            if self.save_trajectory and len(self.trajectory_log) > 0:
                self.save_trajectory_to_file()

            # Cleanup
            cprint("[Evaluator] Cleaning up...", "yellow")
            self.env.close()
            cprint("[Evaluator] Done!", "green")

    def save_trajectory_to_file(self):
        """Save trajectory log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/real_robot_trajectories")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"trajectory_{timestamp}.npz"

        # Convert to numpy arrays
        np.savez(
            output_file,
            trajectory=self.trajectory_log,
            config=self.config
        )

        cprint(f"[Evaluator] Trajectory saved to {output_file}", "green")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Real robot evaluation for 3D-Diffusion-Policy"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (overrides config)'
    )

    args = parser.parse_args()

    # Load config
    cprint(f"[Main] Loading config from {args.config}", "cyan")
    config = load_config(args.config)

    # Override with command line args
    if args.checkpoint is not None:
        config['model']['checkpoint_path'] = args.checkpoint
        cprint(f"[Main] Using checkpoint: {args.checkpoint}", "yellow")

    if args.device is not None:
        config['model']['device'] = args.device
        cprint(f"[Main] Using device: {args.device}", "yellow")

    # Print config
    cprint("\n[Main] Configuration:", "cyan")
    print(yaml.dump(config, default_flow_style=False))

    # Create evaluator
    evaluator = RealRobotEvaluator(config)

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
