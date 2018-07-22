from tensorforce.agents import PPOAgent
from helper.wrappers import ClientToEnv, DictToListFull, JSONable

class TensorforcePPOAgent:
    def __init__(self, env):
        # Create a Proximal Policy Optimization agent
        self.agent = PPOAgent(
            states=dict(type='float', shape=(347,)),
            actions=dict(type='float', shape=(19,)),
            network=[
                dict(type='dense', size=64),
                dict(type='dense', size=64)
            ],
            batching_capacity=1000,
            step_optimizer=dict(
                type='adam',
                learning_rate=1e-4
            )
        )

        self.env = DictToListFull(env)
    
    def test(self, env):
        # Poll new state from client
        state = self.env.reset(project=False)
        import numpy as np
        # Get prediction from agent, execute
        action = self.agent.act(state)
        obs, rew, done, info = self.env.step(action, project=False)

        # Add experience, agent automatically updates model according to batch size
        self.agent.observe(reward=rew, terminal=done)
