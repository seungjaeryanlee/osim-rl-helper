from .Wrapper import EnvironmentWrapper


class ForceDictObservation(EnvironmentWrapper):
    def __init__(self, env):
        """
        Environment wrapper that wraps local environment to use dict-type
        observation by setting project=False. This can be deprecated once
        the default observation is dict-type rather than list-type.
        """
        super().__init__(env)
        self.env = env
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def step(self, action):
        return self.env.step(action, project=False)
