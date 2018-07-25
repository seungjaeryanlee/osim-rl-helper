class EnvironmentWrapper:
    def __init__(self, env):
        """
        A base template for all environment wrappers.
        """
        self.env = env
        # Attributes
        self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
        self.action_space = env.action_space if hasattr(env, 'action_space') else None
        self.time_limit = env.time_limit if hasattr(env, 'time_limit') else None
        self.submit = env.submit if hasattr(env, 'submit') else None

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
