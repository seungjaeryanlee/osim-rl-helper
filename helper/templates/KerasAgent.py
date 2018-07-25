from .Agent import Agent

class KerasAgent(Agent):
    def __init__(self, observation_space, action_space, filename):
        """
        Template class for agents using Keras RL library.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.filename = filename

    def train(self, env, nb_steps):
        """
        Train agent for nb_steps.
        """
        try:
            print('[train] Loading weights from {}'.format(self.filename))
            self.agent.load_weights(self.filename)
            print('[train] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[train] Pretrained model {} not found. Starting from scratch.'.format(self.filename))

        print('[train] Training \'{}\''.format(type(self).__name__))
        self.agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1,
                       nb_max_episode_steps=env.time_limit, log_interval=1000)
        print('[train] Finished training')

        print('[train] Saved weights to \'{}\''.format(self.filename))
        self.agent.save_weights(self.filename, overwrite=True)
        print('[train] Successfully saved weights to \'{}\''.format(self.filename))

    def test(self, env):
        """
        Run agent locally.
        """
        try:
            print('[test] Loading weights from {}'.format(self.filename))
            self.agent.load_weights(self.filename)
            print('[test] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[test] Unable to find pretrained model {}. Aborting.'.format(self.filename))
            return

        print('[test] Running \'{}\''.format(type(self).__name__))
        self.agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
        print('[test] Finished test')

    def submit(self, env):
        """
        Submit agent to CrowdAI server.
        """
        try:
            print('[submit] Loading weights from {}'.format(self.filename))
            self.agent.load_weights(self.filename)
            print('[submit] Successfully loaded weights from {}'.format(self.filename))
        except OSError:
            print('[submit] Unable to find pretrained model {}. Aborting.'.format(self.filename))
            return

        print('[submit] Running \'{}\''.format(type(self).__name__))
        try:
            self.agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=500)
        except TypeError:
            # When observation is None - no more steps left
            pass
        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')
