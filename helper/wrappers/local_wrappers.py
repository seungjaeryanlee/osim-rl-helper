from .Wrapper import EnvironmentWrapper


class ForceDictObservation(EnvironmentWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def step(self, action):
        return self.env.step(action, project=False)
