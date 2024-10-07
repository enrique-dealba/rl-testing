import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.utils.initialization import layer_init


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Try to get observation space, handling both old and new versions
        obs_space = (
            getattr(envs, "single_observation_space", None) or envs.observation_space
        )
        act_space = getattr(envs, "single_action_space", None) or envs.action_space

        # note: Observation space shape, (C, H, W)
        obs_shape = obs_space.shape
        input_channels = obs_shape[0]
        n_actions = act_space.n

        assert input_channels in [
            1,
            3,
            4,
        ], f"Expected 1, 3, or 4 channels, got {input_channels}"

        # Define convolutional layers
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        # Compute the output size of the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            conv_output = self.conv(dummy_input)
            conv_output_size = conv_output.numel()

        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(conv_output_size, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward_conv(self, x):
        # Normalize if input is image data; adjust as needed
        return self.conv(x / 255.0)

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.fc(x)
        return x

    def get_value(self, x):
        hidden = self.forward(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.forward(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
