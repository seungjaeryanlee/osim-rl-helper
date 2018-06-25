# Helper for NIPS 2018: AI for Prosthetics

This is a small repository containing a `helper` package that contains templates and small snipplets of code to help you start the [AI for Prosthetics](https://github.com/stanfordnmbl/osim-rl) challenge.

## How to Start

`example.py` contains an example of how to use the helper package.

```python
import argparse
from osim.env import ProstheticsEnv

from helper.template import Agent
```

The `helper.template.Agent` is a template class for agents. It has two functions implemented:

 * `test()` runs the agent locally.
 * `submit()` submits the agent to the CrowdAI server.

Both `test()` and `submit()` use `Agent.act()` to select action, so you need to implement it. Check the `RandomAgent.act()` for an example.

```python
class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env

    def act(self, observation):
        return self.env.action_space.sample().tolist()
```

The `RandomAgent` class inherits `helper.template.Agent` and overrides `__init__()` and `act()`.

**It seems like the server (or the communication to server) cannot handle NumPy, so the action should be converted to native Python with `.tolist()`.**

```python
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
        agent.test(env)
```

To help you run agent locally or submit agent easily, we used an argument parser. You can run the agent locally with the following command:

```bash
python3 example.py
```

To submit the agent, you should first go to `CONFIG.py` and add your `crowdai_token`. Then, you can submit the agent to the server with the following command:

```bash
python3 example.py -s
```

## FAQ

### Where can I find my CrowdAI token?

Go to this link with `[username]` replaced with your actual CrowdAI username:

```
https://www.crowdai.org/participants/[username]
```

You should see the text `API key: XXXX`. That is your token!
