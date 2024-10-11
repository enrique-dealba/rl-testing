import io

import gym
import streamlit as st
from pettingzoo.mpe import simple_adversary_v3
from PIL import Image

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
            env = simple_adversary_v3.parallel_env(render_mode="rgb_array")
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
