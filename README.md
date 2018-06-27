# Helper for NIPS 2018: AI for Prosthetics

This is a small repository containing a `helper` package that contains a few baseline agents to help you start the [AI for Prosthetics](https://github.com/stanfordnmbl/osim-rl) challenge.



## How to Start

The package contains `run.py` that can test or submit any agent in `/helper/baselines` or `/agent` directory. For example, to test `RandomAgent` locally, run

```bash
./run.py RandomAgent
```

To test `RandomAgent` locally with visualization, run,

```python
./run.py RandomAgent -v
```

To submit `RandomAgent` to CrowdAI, run

```python
./run.py RandomAgent -s
```

Note that you need to first add your API token to `CONFIG.py` to submit any agents. Also note that you can only submit 5 times each 24 hours.



## Create an Agent

The `/agent` directory contains `DoNothingAgent` to serve as an example for custom agents.

```python
from helper.template import Agent


class DoNothingAgent(Agent):
    """
    An agent that chooses NOOP action at every timestep.
    """
    def __init__(self, env):
        self.action = [0] * 19

    def act(self, observation):
        return self.action
```

To be compatible with `run.py`, the agent must inherit from `Agent` from `helper.template`. The agent must also define `.act()` function that returns an action given an observation.



## FAQ

### Where can I find my CrowdAI token?

Go to this link with `[username]` replaced with your actual CrowdAI username:

```
https://www.crowdai.org/participants/[username]
```

You should see the text `API key: XXXX`. That is your token!

