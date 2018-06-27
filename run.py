#!/usr/bin/env python3
"""
You can run RandomAgent locally with following command:

    ./run.py RandomAgent

You can run FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -v

You can submit FixedActionAgent with visuals with following command:

    ./run.py FixedActionAgent -s
"""
import argparse
from osim.env import ProstheticsEnv

from helper.baselines import *
from agents import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run or submit agent.')
    parser.add_argument('agent', help='specify agent\'s class name.')
    parser.add_argument('-s', '--submit', action='store_true', default=False,
                        help='submit agent to crowdAI server')
    parser.add_argument('-v', '--visualize', action='store_true', default=False,
                        help='render the environment locally')
    args = parser.parse_args()

    if args.agent not in globals():
        raise ValueError('[run] Agent {} not found.'.format(args.agent))
    SpecifiedAgent = globals()[args.agent]

    if args.submit:
        # Submit agent
        env = ProstheticsEnv(visualize=False)
        agent = SpecifiedAgent(env)
        agent.submit()
    else:
        # Run agent locally
        env = ProstheticsEnv(visualize=args.visualize)
        agent = SpecifiedAgent(env)
        agent.test(env)
