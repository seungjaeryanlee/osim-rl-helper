import numpy as np


class ClientToEnv:
    def __init__(self, client):
        """
        Reformats client environment to a local environment format.
        """
        self.reset = client.env_reset
        self.step  = client.env_step


class JSONable:
    def __init__(self, env):
        """
        Converts NumPy ndarray type actions to list.
        """
        self.env = env
        self.reset = self.env.reset

    def step(self, action):
        if type(action) == np.ndarray:
            return self.env.step(action.tolist())
        else:
            return self.env.step(action)
