# Helper for NIPS 2018: AI for Prosthetics

This is a helper package designed to help you start the [AI for Prosthetics](https://github.com/stanfordnmbl/osim-rl) challenge.



## How to Start

The package contains `run.py` that can train, test or submit any agent in `/helper/baselines` or `/agent` directory. For example, to train `TensorforcePPOAgent` for 1000 steps, run

```bash
./run.py TensorforcePPOAgent --train 1000
```

After you trained the agent sufficiently, test the agent locally.

```bash
./run.py TensorforcePPOAgent
```

You can also test the agent with visualization with the `-v/--visualize` flag.

```bash
./run.py TensorforcePPOAgent -v
```

If you are satisfied with the result, you can submit the agent to CrowdAI with the `-s/--submit` flag.

```bash
./run.py RandomAgent -s
```

Note that you need to first add your API token to `CONFIG.py` to submit any agents. Also note that you can only submit 5 times each 24 hours ([Issue #141](https://github.com/stanfordnmbl/osim-rl/issues/141)).



## Baseline Agents

### Basic Agents

To understand the environment, consider running the most basic agents. There are two non-learning baseline agents: `RandomAgent` and `FixedActionAgent`. The `RandomAgent` chooses a random action at every timestep, and the `FixedActionAgent` chooses the same action at every timestep. Try running the agent locally to gain some intuition about the environment and the competition.

### KerasDDPGAgent

The `KerasDDPGAgent` uses the Deep Deterministic Policy Gradient algorithm by [Lillicrap et al. (2015)](https://arxiv.org/abs/1509.02971). To use this agent, you need the `keras-rl` package.

### TensorforcePPOAgent

The `TensorforcePPOAgent` uses the Proximal Policy Optimization algorithm by [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347). To use this agent, you need the `tensorforce` package.

### Create a Custom Agent

You can add custom agents to the  `/agent` directory. The directory contains `DoNothingAgent` to serve as an example for custom agents. All agents in the `/agents` directory is imported in `./run.py`, so you can use the same commands as above. If you would like to change the network architecture or hyperparameters of the `keras-rl` or `tensorforce` agents, you can also copy the baseline agent class to this directory and modify it.

```python
from helper.templates import Agent


class DoNothingAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, observation_space, action_space):
        self.action = [0] * action_space.shape[0]

    def act(self, observation):
        return self.action
```

## Where can I find more information?

I am writing a post every week about the competition in [my blog](https://www.endtoend.ai/blog).

 * [Understanding the Challenge](https://www.endtoend.ai/blog/ai-for-prosthetics-1)
 * [Understanding the Action Space](https://www.endtoend.ai/blog/ai-for-prosthetics-2)
 * [Understanding the Observation Space](https://www.endtoend.ai/blog/ai-for-prosthetics-3)

Of course, you should always check the official page for updates.

 * [CrowdAI page](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)
 * [GitHub repository](https://github.com/stanfordnmbl/osim-rl)

If you have any questions about the competition, ask them in [Gitter](https://gitter.im/crowdAI/NIPS-Learning-To-Run-Challenge) or [CrowdAI discussion page](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge/topics)!

## FAQ

### Where can I find my CrowdAI token?

Go to this link with `[username]` replaced with your actual CrowdAI username:

```
https://www.crowdai.org/participants/[username]
```

You should see the text `API key: XXXX`. That is your token!

### I'm getting an 400 Server Error saying that I have no submission slots remaining.

According to [Issue #141](https://github.com/stanfordnmbl/osim-rl/issues/141), you are limited to 5 submissions in a 24-hour window. It seems like the counter is incremented in the beginning when you call `client.env_create()`, so make sure your code is working before attempting to submit!

