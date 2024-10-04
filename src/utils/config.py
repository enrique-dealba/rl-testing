import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="RL Training Script")

    # Wandb configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "rl-project"),
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.getenv("WANDB_ENTITY", None),
        help="Wandb entity name",
    )

    # Training configuration
    parser.add_argument(
        "--exp_name", type=str, default="ppo_atari", help="Experiment name"
    )
    parser.add_argument(
        "--env_id", type=str, default="Breakout-v5", help="Gym environment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--torch_deterministic",
        action="store_true",
        help="Ensure deterministic behavior with PyTorch",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--track", action="store_true", help="Track with Wandb")
    parser.add_argument(
        "--total_timesteps", type=int, default=10000000, help="Total timesteps"
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "kron", "mem_saving_kron", "soap"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2.5e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--memory_saving_kron",
        action="store_true",
        help="Use memory saving kron optimizer",
    )

    # Additional training parameters can be added here

    args = parser.parse_args()
    return args
