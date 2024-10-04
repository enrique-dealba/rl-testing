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

    parser.add_argument(
        "--num_envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=128,
        help="Number of steps per environment per update",
    )
    parser.add_argument(
        "--num_minibatches",
        type=int,
        default=4,
        help="Number of minibatches for updates",
    )
    parser.add_argument(
        "--anneal_lr",
        type=bool,
        default=True,
        help="Whether to anneal the learning rate",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="GAE lambda parameter"
    )
    parser.add_argument(
        "--clip_coef", type=float, default=0.2, help="PPO clip coefficient"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--update_epochs", type=int, default=4, help="Number of epochs for each update"
    )
    parser.add_argument(
        "--norm_adv", type=bool, default=True, help="Normalize advantages"
    )
    parser.add_argument("--clip_vloss", type=bool, default=True, help="Clip value loss")
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Target KL divergence for early stopping",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Whether to track the experiment with wandb",
    )

    args = parser.parse_args()
    return args
