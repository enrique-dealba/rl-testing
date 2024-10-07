import io
import os

import gymnasium as gym
import streamlit as st
import torch
import wandb
from PIL import Image

from src.models.agent import Agent


def create_environment(base_env_id):
    versions = ["v5", "v4", "v0"]  # List versions in order of preference
    for version in versions:
        env_id = f"{base_env_id}-{version}"
        try:
            return gym.make(env_id)
        except gym.error.Error:
            continue
    raise ValueError(f"No valid version found for {base_env_id}")


def load_model_from_wandb(run_path, artifact_name="trained-agent-model"):
    """
    Load the trained model from a Wandb artifact.

    Args:
        run_path (str): Wandb run path in the format 'entity/project/run_id'.
        artifact_name (str): Name of the artifact to retrieve.

    Returns:
        agent (Agent): Loaded agent model.
        env (gym.Env): Gym environment instance.
    """
    try:
        # Initialize Wandb with fork start method to handle multiprocessing issues
        wandb.init(
            project=run_path.split("/")[1],
            entity=run_path.split("/")[0],
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )
    except Exception as e:
        st.error(f"Failed to initialize Wandb: {e}")
        raise e

    try:
        # note: Use artifact_name directly, don't include the run_path
        artifact = wandb.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()
    except Exception as e:
        st.error(f"Failed to retrieve artifact '{artifact_name}': {e}")
        raise e

    try:
        # Load the model state_dict
        model_path = os.path.join(artifact_dir, "trained_model.pth")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        st.error(f"Failed to load model from '{model_path}': {e}")
        raise e

    try:
        # Initialize the Agent model
        env = create_environment("MsPacman")
        agent = Agent(env)
        agent.load_state_dict(state_dict)
        agent.eval()
    except ValueError as ve:
        st.error(f"Failed to create environment: {ve}")
        raise
    except Exception as e:
        st.error(f"Failed to initialize the Agent model: {e}")
        raise

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
    try:
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
    except Exception as e:
        st.error(f"Error during episode run: {e}")
        raise e


def main():
    st.title("Reinforcement Learning Agent")

    st.sidebar.header("Wandb Configuration")
    run_path = st.sidebar.text_input(
        "Wandb Run Path",
        value="edealba/rl-gpu-test/ftwcfo67",
    )
    artifact_name = st.sidebar.text_input(
        "Artifact Name", value="trained-agent-model:v0"
    )

    if st.sidebar.button("Load Model and Run Episode"):
        if run_path and artifact_name:
            with st.spinner("Loading model from Wandb..."):
                try:
                    agent, env = load_model_from_wandb(run_path, artifact_name)
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    return

            st.sidebar.write("Running episode...")
            with st.spinner("Running episode..."):
                try:
                    frames = run_episode(agent, env, render=True)
                    st.success("Episode completed!")
                except Exception as e:
                    st.error(f"Error during episode run: {e}")
                    return

            # Display frames as an animated GIF
            if frames:
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
            else:
                st.warning("No frames captured for the episode.")
        else:
            st.error("Please provide both Wandb Run Path and Artifact Name.")


if __name__ == "__main__":
    main()
