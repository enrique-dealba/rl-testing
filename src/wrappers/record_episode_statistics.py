import gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, info = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        """
        info (dict) keys:
        'env_id', 'players', 'lives', 'reward', 'terminated', 'elapsed_step'
        """
        observations, rewards, terminated, truncated, info = super().step(action)
        # self.episode_returns += rewards
        self.episode_returns += info["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        # self.episode_returns *= 1 - terminated
        # self.episode_lengths *= 1 - terminated
        self.episode_returns *= 1 - info["terminated"]
        self.episode_lengths *= 1 - info["terminated"]
        info["r"] = self.returned_episode_returns
        info["l"] = self.returned_episode_lengths
        return (observations, rewards, terminated, truncated, info)
