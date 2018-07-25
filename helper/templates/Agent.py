class Agent:
    """
    Template class for basic agents.
    """
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, observation):
        """
        Return action given an observation from the environment.
        """
        raise NotImplementedError('[train] This agent has no act() function.')

    def train(self, env, nb_steps):
        """
        Train agent for nb_steps.
        """
        raise NotImplementedError('[train] This agent has no train() function.')

    def test(self, env):
        """
        Run agent locally.
        """
        print('[test] Running \'{}\''.format(type(self).__name__))
        observation = env.reset()

        total_reward = 0
        done = False
        while not done:
            action = self.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

        print('[test] Total Reward of \'{}\': {}'.format(type(self).__name__,
                                                        total_reward))

    def submit(self, env):
        """
        Submit agent to CrowdAI server.
        """
        print('[submit] Running \'{}\' on Server environment'.format(type(self).__name__))
        observation = env.reset()
        print('[submit] Setup agent before episode starts')
        self.before_episode()

        episode_count = 1
        step_count = 0
        total_reward = 0
        while True:
            print('[submit] Episode {} Step {}'.format(episode_count, step_count))
            action = self.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print('[submit] Episode {} Total reward: {}'.format(episode_count, total_reward))
                print('[submit] Cleanup agent after episode terminated')
                self.after_episode()
                observation = env.reset()
                print('[submit] Setup agent before episode starts')
                self.before_episode()
                episode_count += 1
                step_count = 0
                total_reward = 0
                if not observation:
                    break
            step_count += 1

        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        env.submit()
        print('[submit] Submitted results successfully!')
