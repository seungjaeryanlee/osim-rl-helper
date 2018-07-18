from helper.templates import Agent


class DoNothingAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, env):
        self.action = [0] * 19

    def act(self, observation):
        return self.action
