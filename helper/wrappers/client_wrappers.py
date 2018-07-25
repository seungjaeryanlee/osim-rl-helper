import gym
import numpy as np

from .Wrapper import EnvironmentWrapper


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


class JSONable(EnvironmentWrapper):
    def __init__(self, env):
        """
        Converts NumPy ndarray type actions to list.
        """
        super().__init__(env)
        self.env = env

    def step(self, action):
        if type(action) == np.ndarray:
            return self.env.step(action.tolist())
        else:
            return self.env.step(action)
