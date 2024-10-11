import streamlit as st

st.set_page_config(page_title="RL Visualization", page_icon="🤖", layout="wide")

st.title("Reinforcement Learning Visualization")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Welcome to the Reinforcement Learning Visualization app!
    
    This app allows you to visualize different RL agents in action.
    
    ### Pages:
    - **Trained Agent**: Visualize a trained agent's performance on Ms. Pacman.
    - **Random Agent**: Visualize a random agent's behavior on selected environments.
    
    Choose a page from the sidebar to get started!
    """
)
