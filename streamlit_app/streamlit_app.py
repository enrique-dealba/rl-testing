import io

import gymnasium as gym
import streamlit as st
import torch
import wandb
from PIL import Image

from src.models.agent import Agent


def load_model_from_wandb(run_path, artifact_name="trained-agent-model"):
    """
    Load the trained model from Wandb artifact.

    Args:
        run_path (str): Wandb run path in the format 'entity/project/run_id'.
        artifact_name (str): Name of the artifact to retrieve.

    Returns:
        agent (Agent): Loaded agent model.
        env (gym.Env): Gym environment instance.
    """
    # Initialize Wandb
    wandb.init(
        project=run_path.split("/")[1],
        entity=run_path.split("/")[0],
        resume="allow",
        anonymous="allow",
    )

    # Retrieve the artifact
    artifact = wandb.use_artifact(f"{run_path}/{artifact_name}:latest", type="model")
    artifact_dir = artifact.download()

    # Load the model state_dict
    model_path = f"{artifact_dir}/trained_model.pth"
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Initialize the Agent model
    env_id = "MsPacman-v5"  # Ensure this matches the environment used during training
    env = gym.make(env_id)
    agent = Agent(env)
    agent.load_state_dict(state_dict)
    agent.eval()

    return agent, env


def run_episode(agent, env, render=True):
    """
    Run a single episode using the trained agent.

    Args:
        agent (Agent): The trained agent.
        env (gym.Env): The Gym environment.
        render (bool): Whether to capture frames for visualization.

    Returns:
        frames (list of PIL.Image): List of frames captured during the episode.
    """
    obs, info = env.reset()
    done = False
    frames = []

    while not done:
        obs_tensor = (
            torch.from_numpy(obs).float().unsqueeze(0) / 255.0
        )  # Normalize if needed
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        action = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render:
            frame = env.render(mode="rgb_array")
            frames.append(Image.fromarray(frame))

    env.close()
    return frames


def main():
    st.title("Trained RL Agent Visualization")

    st.sidebar.header("Wandb Configuration")
    run_path = st.sidebar.text_input(
        "Wandb Run Path", value="my_username/rl-gpu-test/abc123"
    )
    artifact_name = st.sidebar.text_input("Artifact Name", value="trained-agent-model")

    if st.sidebar.button("Load Model and Run Episode"):
        if run_path:
            with st.spinner("Loading model from Wandb..."):
                try:
                    agent, env = load_model_from_wandb(run_path, artifact_name)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    return

            st.sidebar.write("Running episode...")
            with st.spinner("Running episode..."):
                frames = run_episode(agent, env, render=True)
                st.success("Episode completed!")

            # Display frames as an animated GIF
            img_buffer = io.BytesIO()
            frames[0].save(
                img_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
            )
            img_buffer.seek(0)
            st.image(img_buffer, use_column_width=True)


if __name__ == "__main__":
    main()
