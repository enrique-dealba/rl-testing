import io

import gym
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Random Agent", page_icon="🎲")

st.title("Random Agent Visualization")

# List of environments to choose from
ENVIRONMENTS = ["CartPole-v1", "LunarLander-v2", "MountainCar-v0", "Acrobot-v1"]


def run_random_episode(env, max_steps=1000):
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


st.sidebar.header("Environment Configuration")
selected_env = st.sidebar.selectbox("Select Environment", ENVIRONMENTS)

if st.sidebar.button("Run Random Agent"):
    with st.spinner(f"Running random agent on {selected_env}..."):
        env = gym.make(selected_env)
        frames = run_random_episode(env)

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
This page allows you to visualize a random agent's behavior in various OpenAI Gym environments.
Select an environment from the sidebar and click "Run Random Agent" to see the agent in action.
""")
