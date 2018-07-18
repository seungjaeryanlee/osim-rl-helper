from ..templates import Agent


class RandomAgent(Agent):
    """
    An agent that chooses random action at every timestep.
    """
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample().tolist()
