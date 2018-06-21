#!/usr/bin/env python3
import argparse
from osim.env import ProstheticsEnv

from helper.template import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run or submit agent')
    parser.add_argument('-s', '--submit', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.submit:
        # Submit agent
        env = ProstheticsEnv(visualize=False)
        agent = RandomAgent(env)
        agent.submit()
    else:
        # Run agent locally
        env = ProstheticsEnv(visualize=True)
        agent = RandomAgent(env)
        agent.play(env)
