# Baselines

These are simple baseline agents for NIPS 2018: AI for Prosthetics competition.

## Baseline Agents

### Basic Agents

To understand the environment, consider running the most basic agents. There are two non-learning baseline agents: `RandomAgent` and `FixedActionAgent`. The `RandomAgent` chooses a random action at every timestep, and the `FixedActionAgent` chooses the same action at every timestep. Try running the agent locally to gain some intuition about the environment and the competition.

### KerasDDPGAgent

The `KerasDDPGAgent` uses the Deep Deterministic Policy Gradient algorithm by [Lillicrap et al. (2015)](https://arxiv.org/abs/1509.02971). To use this agent, you need the `keras-rl` package.

### TensorforcePPOAgent

The `TensorforcePPOAgent` uses the Proximal Policy Optimization algorithm by [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347). To use this agent, you need the `tensorforce` package.
