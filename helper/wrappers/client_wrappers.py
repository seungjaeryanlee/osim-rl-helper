import gym
import numpy as np


class ClientToEnv:
    def __init__(self, client):
        """
        Reformats client environment to a local environment format.
        """
        self.client = client
        self.reset = client.env_reset
        self.step  = client.env_step
        self.submit = client.submit
        self.time_limit = 300
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(19, ),
                                           dtype=np.float32)


class JSONable:
    def __init__(self, env):
        """
        Converts NumPy ndarray type actions to list.
        """
        self.env = env
        self.reset = self.env.reset
        if hasattr(self.env, 'submit'):
            self.submit = self.env.submit
        if hasattr(self.env, 'observation_space'):
            self.observation_space = self.env.observation_space
        if hasattr(self.env, 'action_space'):
            self.action_space = self.env.action_space
        if hasattr(self.env, 'time_limit'):
            self.time_limit = self.env.time_limit

    def step(self, action):
        if type(action) == np.ndarray:
            return self.env.step(action.tolist())
        else:
            return self.env.step(action)
