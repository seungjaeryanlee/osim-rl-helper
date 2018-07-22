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
        self.load()
        state = self.env.reset(project=False)
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(state)
            obs, rew, done, info = self.env.step(action, project=False)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('Total reward: ' + str(total_rew))
        self.save()
    
    def save(self):
        self.agent.save_model(directory="./TensorforcePPOAgent/")
        print('[save] Saved pretrained model successfully.')

    def load(self):
        self.agent.restore_model(directory="./TensorforcePPOAgent/")
        print('[load] Loaded pretrained model successfully.')
