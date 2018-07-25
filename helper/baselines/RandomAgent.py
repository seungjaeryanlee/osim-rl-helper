from ..templates import Agent

class RandomAgent(Agent):
    """
    An agent that chooses random action at every timestep.
    """

    def act(self, observation):
        return self.action_space.sample().tolist()
