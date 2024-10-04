import os
import random
import time
from collections import deque

import envpool
import numpy as np
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces import Discrete
from torch.utils.tensorboard import SummaryWriter

from src.models.agent import Agent
from src.optimizers.adam_optimizer import get_adam_optimizer
from src.optimizers.kron_optimizer import get_kron_optimizer
from src.optimizers.soap_optimizer import get_soap_optimizer
from src.utils.config import parse_args
from src.utils.logger import setup_logger
from src.wrappers.record_episode_statistics import RecordEpisodeStatistics


def main():
    args = parse_args()
    logger = setup_logger()

    # Error handling for required arguments
    required_args = [
        "num_envs",
        "num_steps",
        "num_minibatches",
        "total_timesteps",
        "anneal_lr",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "update_epochs",
        "norm_adv",
        "clip_vloss",
        "target_kl",
    ]

    for arg in required_args:
        if not hasattr(args, arg):
            raise ValueError(f"ERROR: Required argument '{arg}' is missing.")

    # Set Wandb API key from environment variable
    wandb_key = os.getenv("WANDB_API_KEY")
    if args.track:
        if wandb_key is None:
            raise ValueError(
                "Wandb API key not found. Please set the WANDB_API_KEY environment variable."
            )
        os.environ["WANDB_API_KEY"] = wandb_key
        logger.info("Wandb API key has been set from environment variables.")

    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}",
            monitor_gym=True,
            save_code=True,
        )
        # Construct the run path for logging
        run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
        run_url = wandb.run.get_url()
        logger.info(f"Wandb Run Path: {run_path}")
        logger.info(f"Wandb Run URL: {run_url}")
    else:
        run_path = None
        run_url = None

    writer = SummaryWriter(
        f"runs/{args.exp_name}_{args.env_id}_{args.seed}_{int(time.time())}"
    )
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Environment setup
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    logger.info(f"Action Space Type: {type(envs.action_space)}")
    assert isinstance(
        envs.action_space, Discrete
    ), "Only discrete action space is supported."

    # Initialize Agent
    agent = Agent(envs).to(device)
    agent.train()

    # Log model summary
    logger.info("Model Summary:")
    num_params = 0
    for name, param in agent.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.shape}")
            num_params += param.numel()
    logger.info(f"Number of trainable parameters: {num_params}")

    # Initialize Optimizer
    if args.optimizer == "adam":
        optimizer = get_adam_optimizer(
            agent.parameters(), args.learning_rate, args.weight_decay
        )
    elif args.optimizer in ["kron", "mem_saving_kron"]:
        optimizer = get_kron_optimizer(
            agent.parameters(),
            args.learning_rate,
            args.weight_decay,
            memory_saving=(args.optimizer == "mem_saving_kron"),
        )
    elif args.optimizer == "soap":
        optimizer = get_soap_optimizer(
            agent.parameters(), args.learning_rate, args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    logger.info(f"Using optimizer: {args.optimizer}")

    # Training loop setup
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    logger.info(f"Starting training for {num_iterations} iterations.")

    avg_returns = deque(maxlen=20)

    global_step = 0
    start_time = time.time()
    # next_obs = torch.Tensor(envs.reset()).to(device)
    # After resetting the environment
    next_obs_np, _ = envs.reset()
    # Ensure the observations are float tensors and move them to the correct device
    next_obs = torch.from_numpy(next_obs_np).float().to(device)

    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, num_iterations + 1):
        # Learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1) / num_iterations
            lr_now = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        # Collect experiences
        agent.eval()
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        ).to(device)
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape
        ).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # next_obs_np, reward, next_done_np, info = envs.step(action.cpu().numpy())
            # rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs = torch.from_numpy(next_obs_np).to(device)
            # next_done = torch.from_numpy(next_done_np).to(device).float()
            next_obs_np, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.from_numpy(next_obs_np).float().to(device)
            next_done = torch.from_numpy(done).to(device).float()

            # Verify shapes of observations match what Agent network expects
            # logger.info(
            #     f"Observation Space Shape: {envs.single_observation_space.shape}"
            # )
            # logger.info(f"Sample Observation Shape: {next_obs.shape}")

            # Logging
            for idx, done_flag in enumerate(next_done):
                if done_flag and info["l"][idx] == 0:
                    if iteration % 20 == 0:
                        logger.info(
                            f"global_step={global_step}, episodic_return={info['r'][idx]}"
                        )
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar(
                        "charts/avg_episodic_return", np.mean(avg_returns), global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["r"][idx], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["l"][idx], global_step
                    )

        # Compute advantages and returns
        agent.train()
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                logger.info(f"Early stopping at epoch {epoch} due to reaching max kl.")
                break

        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            1 - np.var(y_true - y_pred) / var_y if var_y > 0 else float("nan")
        )

        # Logging
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if args.track:
            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "value_loss": v_loss.item(),
                    "policy_loss": pg_loss.item(),
                    "entropy": entropy_loss.item(),
                    "old_approx_kl": old_approx_kl.item(),
                    "approx_kl": approx_kl.item(),
                    "clipfrac": np.mean(clipfracs),
                    "explained_variance": explained_var,
                    "SPS": int(global_step / (time.time() - start_time)),
                }
            )

    # Save the trained model
    model_path = "trained_model.pth"
    torch.save(agent.state_dict(), model_path)
    logger.info(f"Saved trained model to {model_path}")

    if args.track:
        # Create a Wandb Artifact
        artifact = wandb.Artifact("trained-agent-model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        logger.info("Uploaded model to Wandb as an artifact.")

        # Log artifact details
        logger.info(f"Artifact Name: {artifact.name}")
        logger.info(f"Artifact Type: {artifact.type}")
        logger.info(f"Artifact Version: {artifact.version}")
        logger.info(f"Wandb Run Path: {run_path}")

        # Optionally, log model parameters and summary
        model_summary = {
            "num_trainable_params": sum(
                p.numel() for p in agent.parameters() if p.requires_grad
            ),
            "model_architecture": str(agent),
        }
        wandb.config.update(model_summary)
        logger.info("Logged model summary and parameters to Wandb.")

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
