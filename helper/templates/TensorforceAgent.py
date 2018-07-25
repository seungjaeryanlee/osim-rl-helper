from .Agent import Agent


class TensorforceAgent(Agent):
    def __init__(self, observation_space, action_space, directory):
        """
        Template class for agents using Keras RL library.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.directory = directory
        self.agent = None

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
        """
        Submit agent to CrowdAI server.
        """
        try:
            print('[submit] Loading weights from \'{}\''.format(self.directory))
            self.agent.restore_model(directory=self.directory)
            print('[submit] Successfully loaded weights from \'{}\''.format(self.directory))
        except ValueError:
            print('[submit] Unable to find pretrained model from \'{}\'. Aborting.'.format(self.directory))
            return

        print('[submit] Running \'{}\''.format(type(self).__name__))
        obs = env.reset()
        episode_count = 1
        step_count = 0
        total_rew = 0
        try:
            while True:
                action = self.act(obs)
                obs, rew, done, info = env.step(action)
                total_rew += rew
                step_count += 1
                if done:
                    print('[submit] Episode {} | Steps Taken: {:3} | Total reward: {}'.format(episode_count, step_count, total_rew))
                    obs = env.reset()
                    episode_count += 1
                    step_count = 0
                    total_rew = 0
        except TypeError:
            # When observation is None - no more steps left
            pass

        print('[submit] Finished running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')

    def act(self, obs):
        return self.agent.act(obs)
