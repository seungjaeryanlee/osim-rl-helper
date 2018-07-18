from ..templates import Agent


class FixedActionAgent(Agent):
    """
    An agent that choose one fixed action at every timestep.
    """
    def __init__(self, env):
        self.action = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]

    def act(self, observation):
        return self.action
