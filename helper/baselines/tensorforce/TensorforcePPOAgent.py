from tensorforce.agents import PPOAgent
from helper.wrappers import ClientToEnv, DictToListFull, JSONable


class TensorforcePPOAgent:
    def __init__(self, observation_space, action_space,
                 directory='./TensorforcePPOAgent/'):
        # Create a Proximal Policy Optimization agent
        self.agent = PPOAgent(
            states=dict(type='float', shape=observation_space.shape),
            actions=dict(type='float', shape=action_space.shape,
                         min_value=0, max_value=1),
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
        self.directory = directory

    def train(self, env, nb_steps):
        try:
            print('[train] Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[train] Successfully loaded weights from {}'.format(self.directory))
        except ValueError:
            print('[train] Pretrained model {} not found. Starting from scratch.'.format(self.directory))

        print('[train] Training \'{}\''.format(type(self).__name__))
        step_count = 0
        episode_count = 1
        while step_count < nb_steps:
            episode_step_count = 0
            obs = env.reset()
            done = False
            total_rew = 0
            while not done:
                action = self.agent.act(obs)
                obs, rew, done, info = env.step(action)
                total_rew += rew
                self.agent.observe(reward=rew, terminal=done)
                episode_step_count += 1
            step_count += episode_step_count
            print('[train] Episode {:3} | Steps Taken: {:3} | Total Steps: Taken {:6}/{:6} | Total reward: {}'.format(
                episode_count, episode_step_count, step_count, nb_steps, total_rew))
            episode_count += 1
        print('[train] Finished training')

        print('[train] Saved weights to \'{}\''.format(self.directory))
        self.agent.save_model(directory=self.directory)
        print('[train] Successfully saved weights to \'{}\''.format(self.directory))

    def test(self, env):
        """
        Run agent locally.
        """
        try:
            print('[test] Loading weights from {}'.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[test] Successfully loaded weights from {}'.format(self.directory))
        except ValueError:
            print('[test] Unable to find pretrained model {}. Aborting.'.format(self.directory))
            return

        print('[test] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        done = False
        total_rew = 0
        while not done:
            action = self.agent.act(obs)
            obs, rew, done, info = env.step(action)
            total_rew += rew
            self.agent.observe(reward=rew, terminal=done)
        print('[test] Total reward: ' + str(total_rew))
        print('[test] Finished test.')

        print('[test] Saved weights to \'{}\''.format(self.directory))
        self.agent.save_model(directory=self.directory)
        print('[test] Successfully saved weights to \'{}\''.format(self.directory))

    def submit(self, env):
        try:
            print('[submit] Loading weights from \'{}\''.format(self.directory))
            self.agent.load_weights(self.directory)
            print('[submit] Successfully loaded weights from \'{}\''.format(self.directory))
        except ValueError:
            print('[submit] Unable to find pretrained model from \'{}\'. Aborting.'.format(self.directory))
            return
        # TODO: Add submit
        return
