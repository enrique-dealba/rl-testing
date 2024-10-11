import io
import os

import envpool
import numpy as np
import streamlit as st
import torch
import wandb
from PIL import Image

from src.models.agent import Agent
from src.wrappers.record_episode_statistics import RecordEpisodeStatistics

st.set_page_config(page_title="Trained Agent", page_icon="🎮")

st.title("Reinforcement Learning Agent")


def create_environment(env_id, num_envs=1):
    envs = envpool.make(
        env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        episodic_life=True,  # This was already set in your original code
        reward_clip=True,  # This was already set in your original code
        stack_num=4,  # Default stack_num, matching your likely training setup
        gray_scale=True,  # Default setting
        frame_skip=4,  # Default frame_skip
        img_height=84,  # Default image height
        img_width=84,  # Default image width
        noop_max=30,  # Default noop_max
        seed=42,  # TODO: make this configurable
    )
    envs.num_envs = num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    return envs


def load_model_from_wandb(run_path, artifact_name="trained-agent-model"):
    try:
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
        model_path = os.path.join(artifact_dir, "trained_model.pth")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        st.error(f"Failed to load model from '{model_path}': {e}")
        raise e

    try:
        envs = create_environment("MsPacman-v5")
        agent = Agent(envs).to("cpu")
        agent.load_state_dict(state_dict)
        agent.eval()
    except Exception as e:
        st.error(f"Failed to initialize the Agent model: {e}")
        raise e

    return agent, envs


def run_episode(agent, envs, render=True):
    try:
        next_obs, _ = envs.reset()
        next_obs = torch.from_numpy(next_obs).float()
        done = False
        frames = []

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(next_obs)
            action = action.cpu().numpy()
            next_obs, _, terminated, truncated, _ = envs.step(action)
            next_obs = torch.from_numpy(next_obs).float()
            done = np.logical_or(terminated, truncated)[0]

            if render:
                # Use the last frame of the stacked observation
                frame = next_obs[0, -1].cpu().numpy()  # Shape should be (84, 84)
                # Convert to RGB by repeating the channel
                frame_rgb = np.stack([frame] * 3, axis=-1)
                # Normalize to 0-255 range
                frame_rgb = (frame_rgb * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame_rgb))

        return frames
    except Exception as e:
        st.error(f"Error during episode run: {e}")
        raise e


st.sidebar.header("Wandb Configuration")

# Initialize session state for run_path if doesn't exist
if "run_path" not in st.session_state:
    st.session_state.run_path = "edealba/rl-gpu-test/ftwcfo67"

# Use session state for text input
run_path = st.sidebar.text_input("Wandb Run Path", key="run_path")

# Initialize session state for artifact_name if doesn't exist
if "artifact_name" not in st.session_state:
    st.session_state.artifact_name = "trained-agent-model:v0"

# Use session state for text input
artifact_name = st.sidebar.text_input("Artifact Name", key="artifact_name")

st.write(f"Current Run Path: {run_path}")
st.write(f"Current Artifact Name: {artifact_name}")

if st.sidebar.button("Load Model and Run Episode"):
    if run_path and artifact_name:
        with st.spinner("Loading model from Wandb..."):
            try:
                agent, envs = load_model_from_wandb(run_path, artifact_name)
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        st.sidebar.write("Running episode...")
        with st.spinner("Running episode..."):
            try:
                frames = run_episode(agent, envs, render=True)
                st.success("Episode completed!")
            except Exception as e:
                st.error(f"Error during episode run: {e}")
                st.stop()

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
