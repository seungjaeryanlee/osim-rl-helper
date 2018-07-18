import json

from osim.http.client import Client

from ..wrappers import ClientToEnv, DictToListFull, JSONable
from ..CONFIG import remote_base, crowdai_token


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

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False