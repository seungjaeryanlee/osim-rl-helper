import json
from osim.http.client import Client

from .CONFIG import remote_base, crowdai_token


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

class Agent:
    def __init__(self):
        pass

    def act(self, observation):        
        pass

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
        total_reward = 0
        done = False
        while not done:
            action = self.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

        print('[test] Total Reward of \'{}\': {}'.format(type(self).__name__,
                                                        total_reward))

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
                observation = client.env_reset()
                episode_count += 1
                step_count = 0
                total_reward = 0
                if not observation:
                    break
            step_count += 1

        print('[submit] Finished Running \'{}\' on Server environment. Submitting results to server...'.format(type(self).__name__))
        client.submit()
        print('[submit] Submitted results successfully!')
