import json

from osim.http.client import Client

from .wrappers import ClientToEnv, DictToListLegacy, JSONable
from .CONFIG import remote_base, crowdai_token


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

class Agent:
    """
    Template class for basic agents.
    """
    def __init__(self):
        pass

    def act(self, observation):
        """
        Return action given an observation from the environment.
        """
        pass

    def before_episode(self):
        """
        Run before the start of every episode.
        """
        pass

    def after_episode(self):
        """
        Run after the end of every episode.
        """
        pass

    def train(self, env, nb_steps):
        """
        Train agent for nb_steps.
        """
        raise NotImplementedError('[train] This agent\'s train() function is not implemented.')

    def test(self, env):
        """
        Run agent locally.
        """
        error, message = self.sanity_check()
        if error:
            print('[test] Sanity check failed')
            print(message)
            return

        print('[test] Sanity check passed')
        print('[test] Running \'{}\''.format(type(self).__name__))
        observation = env.reset()
        print('[test] Setup agent before episode starts')
        self.before_episode()
        total_reward = 0
        done = False
        while not done:
            action = self.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

        print('[test] Total Reward of \'{}\': {}'.format(type(self).__name__,
                                                        total_reward))

        print('[test] Cleanup agent after episode terminated')
        self.after_episode()

    def sanity_check(self):
        """
        Check if the agent's actions are legal.
        """
        observation = [0] * 158
        action = self.act(observation)
        if type(action) is not list:
            return (True, 'Action should be a list: are you using NumPy?')
        if not is_jsonable(action):
            return (True, 'Action should be jsonable: are you using NumPy?')

        return (False, '')

    def submit(self):
        """
        Submit agent to CrowdAI server.
        """
        error, message = self.sanity_check()
        if error:
            print('[submit] Sanity check failed')
            print(message)
            return

        print('[submit] Sanity check passed')
        print('[submit] Running \'{}\' on Server environment'.format(type(self).__name__))
        client = Client(remote_base)
        observation = client.env_create(crowdai_token, env_id='ProstheticsEnv')
        print('[submit] Setup agent before episode starts')
        self.before_episode()

        episode_count = 1
        step_count = 0
        total_reward = 0
        while True:
            print('[submit] Episode {} Step {}'.format(episode_count, step_count))
            action = self.act(observation)
            observation, reward, done, info = client.env_step(action)
            total_reward += reward
            if done:
                print('[submit] Episode {} Total reward: {}'.format(episode_count, total_reward))
                print('[submit] Cleanup agent after episode terminated')
                self.after_episode()
                observation = client.env_reset()
                print('[submit] Setup agent before episode starts')
                self.before_episode()
                episode_count += 1
                step_count = 0
                total_reward = 0
                if not observation:
                    break
            step_count += 1

        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        client.submit()
        print('[submit] Submitted results successfully!')


class KerasAgent:
    def __init__(self):
        """
        Template class for agents using Keras RL library.
        """
        self.agent = None
        self.filename = None

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

    def submit(self):
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

        print('[submit] Connecting to client...')
        client = Client(remote_base)
        client.env_create(crowdai_token, env_id='ProstheticsEnv')
        print('[submit] Successfully connected!')

        env = ClientToEnv(client)
        env = DictToListLegacy(env)
        env = JSONable(env)

        print('[submit] Running \'{}\''.format(type(self).__name__))
        try:
            self.agent.test(env, nb_episodes=3, visualize=False, nb_max_episode_steps=500)
        except TypeError:
            # When observation is None - no more steps left
            pass
        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        client.submit()
        print('[submit] Submitted results successfully!')
