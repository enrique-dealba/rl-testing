import io

import gym
import pettingzoo
import streamlit as st
from PIL import Image

st.write(f"PettingZoo version: {pettingzoo.__version__}")

possible_envs = [
    "simple_adversary_v3",
    "simple_crypto_v2",
    "simple_push_v2",
    "simple_reference_v2",
    "simple_speaker_listener_v3",
    "simple_spread_v3",
    "simple_tag_v2",
    "simple_world_comm_v2",
    "simple_adversary_v1",
    "simple_crypto_v1",
    "simple_push_v1",
    "simple_reference_v1",
    "simple_speaker_listener_v1",
    "simple_spread_v1",
    "simple_tag_v1",
    "simple_world_comm_v1",
    "simple_adversary_v2",
]

available_envs = []

for env in possible_envs:
    try:
        # Dynamically import the environment
        module = __import__(f"pettingzoo.mpe.{env}", fromlist=[""])
        available_envs.append(env)
    except ImportError:
        # Skip if import fails
        pass

# Display the available environments in Streamlit
st.write("Available environments:", available_envs)

from pettingzoo.mpe import simple_adversary_v2

st.set_page_config(page_title="Random Agent", page_icon="🎲")

st.title("Random Agent Visualization")

# List of environments to choose from
GYM_ENVIRONMENTS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
MPE_ENVIRONMENTS = ["simple_adversary_v3"]


def run_random_episode_gym(env, max_steps=1000):
    obs = env.reset()
    done = False
    frames = []
    for _ in range(max_steps):
        frame = env.render(mode="rgb_array")
        frames.append(Image.fromarray(frame))
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            break
    env.close()
    return frames


def run_random_episode_mpe(env, max_steps=1000):
    env.reset()
    frames = []
    for _ in range(max_steps):
        frame = env.render()
        frames.append(Image.fromarray(frame))
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            break
    env.close()
    return frames


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
            frames = run_random_episode_gym(env)
        else:
            env = simple_adversary_v2.parallel_env(render_mode="rgb_array")
            frames = run_random_episode_mpe(env)

        if frames:
            st.success("Episode completed!")
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
