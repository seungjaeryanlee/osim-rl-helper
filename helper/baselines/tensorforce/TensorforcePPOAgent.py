from tensorforce.agents import PPOAgent
from helper.wrappers import ClientToEnv, DictToListFull, JSONable

class TensorforcePPOAgent:
    def __init__(self, env, directory=./'TensorforcePPOAgent/'):
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
    
    def train(self, env, nb_steps):
        try:
            print('[train] Loading weights from {}'.format(self.filename))
            self.agent.restore_model(directory=directory)
            print('[train] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[train] Pretrained model {} not found. Starting from scratch.'.format(self.filename))

        print('[train] Training \'{}\''.format(type(self).__name__))
        for step in nb_steps:
        state = self.env.reset(project=False)
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(state)
            obs, rew, done, info = self.env.step(action, project=False)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('[train] Finished training')

        print('[train] Saved weights to \'{}\''.format(self.filename))
        self.agent.save_model(directory=directory)
        print('[train] Successfully saved weights to \'{}\''.format(self.filename))

    def test(self, env):
        """
        Run agent locally.
        """
        try:
            print('[test] Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=directory)
            print('[test] Successfully loaded weights from {}'.format(self.directory))
        except OSError:
            print('[test] Unable to find pretrained model {}. Aborting.'.format(self.filename))
            return

        print('[test] Running \'{}\''.format(type(self).__name__))
        state = self.env.reset(project=False)
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(state)
            obs, rew, done, info = self.env.step(action, project=False)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('[test] Total reward: ' + str(total_rew))
        print('[test] Finished test.')

        print('[test] Saved weights to \'{}\''.format(self.filename))
        self.agent.save_model(directory=directory)
        print('[test] Successfully saved weights to \'{}\''.format(self.filename))

    def submit(self, env):
        try:
            print('[submit] Loading weights from {}'.format(self.filename))
            self.agent.load_weights(self.filename)
            print('[submit] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[submit] Unable to find pretrained model {}. Aborting.'.format(self.filename))
            return
        # TODO: Add submit
        return
