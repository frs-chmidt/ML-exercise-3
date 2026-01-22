"""
episode.py

Helpers to:
1) generate an episode (sequence of transitions) from a policy
2) compute Monte Carlo returns G_t for each time step
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from grid import env, state, action

# One transition in an episode: (state, action, reward)
Transition = Tuple[state, action, int]


def generate_episode_random(e: env, max_steps: int = 500) -> List[Transition]:
    """
    Generate ONE episode using a random policy.

    Episode = list of (state, action, reward).
    Stops when:
      - environment signals done=True, OR
      - max_steps is reached
    """
    episode: List[Transition] = []
    s = e.reset()

    # Local RNG (cleaner than using np.random global state)
    rng = np.random.default_rng()

    for _ in range(max_steps):
        # Pick a random action from the action space
        a = e.action_space[rng.integers(len(e.action_space))]

        s_next, r, done, _info = e.step(a)

        # Store transition: current state -> action -> reward
        episode.append((s, a, r))

        # Move forward
        s = s_next

        if done:
            break

    return episode


def compute_returns(episode: List[Transition], gamma: float = 1.0) -> List[float]:
    """
    Compute Monte Carlo returns for an episode.

    Return at time t:
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Output list has same length as episode.
    """
    returns = [0.0] * len(episode)

    G = 0.0
    # Walk backwards so we can accumulate G efficiently
    for t in range(len(episode) - 1, -1, -1):
        _s, _a, r = episode[t]
        G = r + gamma * G
        returns[t] = G

    return returns
