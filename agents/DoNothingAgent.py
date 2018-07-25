from helper.templates import Agent


class DoNothingAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, observation_space, action_space):
        self.action = [0] * action_space.shape[0]

    def act(self, observation):
        return self.action
