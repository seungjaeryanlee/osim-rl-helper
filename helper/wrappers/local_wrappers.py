class ForceDictObservation:
    def __init__(self, env):
        self.env = env
        if hasattr(self.env, 'submit'):
            self.submit = self.env.submit
        if hasattr(self.env, 'action_space'):
            self.action_space = self.env.action_space
        self.time_limit = 300
    
    def reset(self):
        return self.env.reset(project=False)

    def step(self, action):
        return self.env.step(action, project=False)
