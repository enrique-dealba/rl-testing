import io

import gym
import pettingzoo
import streamlit as st
from pettingzoo.mpe import (
    simple_adversary_v2,
    simple_crypto_v2,
    simple_push_v2,
    simple_reference_v2,
    simple_speaker_listener_v3,
    simple_tag_v2,
    simple_world_comm_v2,
)
from PIL import Image

st.set_page_config(page_title="Random Agent", page_icon="🎲")
st.write(f"PettingZoo version: {pettingzoo.__version__}")


def run_random_episode_mpe(env, max_steps=1000, render=True):
    env.reset()
    frames = []
    total_steps = 0
    episode_done = False

    while not episode_done and total_steps < max_steps:
        for agent in env.agent_iter():
            total_steps += 1
            last_values = env.last()
            if len(last_values) == 5:
                observation, reward, termination, truncation, info = last_values
            elif len(last_values) == 4:
                observation, reward, done, info = last_values
                termination = truncation = done
            else:
                raise ValueError(
                    f"Unexpected number of values from env.last(): {len(last_values)}"
                )

            if termination or truncation:
                action = None
                episode_done = True
            else:
                action = env.action_space(agent).sample()

            env.step(action)

        if render:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
            except Exception as e:
                st.warning(f"Rendering not supported: {e}")
                render = False

        if episode_done:
            break

    env.close()
    return frames, total_steps


def run_random_episode_gym(env, max_steps=1000):
    obs = env.reset()
    frames = []
    total_steps = 0
    done = False

    while not done and total_steps < max_steps:
        total_steps += 1
        frame = env.render(mode="rgb_array")
        frames.append(Image.fromarray(frame))
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    return frames, total_steps


def create_mpe_env(env_name):
    env_mapping = {
        "simple_adversary_v2": simple_adversary_v2,
        "simple_crypto_v2": simple_crypto_v2,
        "simple_push_v2": simple_push_v2,
        "simple_reference_v2": simple_reference_v2,
        "simple_speaker_listener_v3": simple_speaker_listener_v3,
        "simple_tag_v2": simple_tag_v2,
        "simple_world_comm_v2": simple_world_comm_v2,
    }
    return env_mapping[env_name].env(render_mode="rgb_array")


st.title("Random Agent Visualization")

# List of environments to choose from
GYM_ENVIRONMENTS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
MPE_ENVIRONMENTS = [
    "simple_adversary_v2",
    "simple_crypto_v2",
    "simple_push_v2",
    "simple_reference_v2",
    "simple_speaker_listener_v3",
    "simple_tag_v2",
    "simple_world_comm_v2",
]

st.sidebar.header("Environment Configuration")
env_type = st.sidebar.radio("Environment Type", ["Gym", "MPE"])

if env_type == "Gym":
    selected_env = st.sidebar.selectbox("Select Gym Environment", GYM_ENVIRONMENTS)
else:
    selected_env = st.sidebar.selectbox("Select MPE Environment", MPE_ENVIRONMENTS)

if st.sidebar.button("Run Random Agent"):
    with st.spinner(f"Running random agent on {selected_env}..."):
        if env_type == "Gym":
            env = gym.make(selected_env)
            frames, total_steps = run_random_episode_gym(env)
        else:
            env = create_mpe_env(selected_env)
            frames, total_steps = run_random_episode_mpe(env)

        if frames:
            st.success(f"Episode completed in {total_steps} steps!")
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

st.markdown("""
This page allows you to visualize a random agent's behavior in various OpenAI Gym and Multi-Particle Environments (MPE).
Select an environment type and specific environment from the sidebar, then click "Run Random Agent" to see the agent(s) in action.
""")
