from __future__ import annotations
from typing import List, Tuple
import numpy as np
from grid import env, state, action

# One transition in an episode: (state, action, reward)
Transition = Tuple[state, action, int]


def generate_episode_random(
    e: env,
    max_steps: int = 500,
) -> List[Transition]:
    """
    Generate ONE episode using a random policy.

    Returns a list of (state, action, reward) transitions.
    Episode ends when done=True OR when max_steps is reached.
    """
    episode: List[Transition] = []

    s = e.reset()

    for _ in range(max_steps):
        # choose a random action from the environment's action space
        a = e.action_space[np.random.randint(len(e.action_space))]

        s_next, r, done, info = e.step(a)

        # store transition (current state, action taken, reward received)
        episode.append((s, a, r))

        # move to next state
        s = s_next

        if done:
            break

    return episode

def compute_returns(episode: List[Transition], gamma: float = 1.0) -> List[float]:
    """
    Given an episode [(s,a,r), ...], compute returns G_t for each time step t.
    Output list has same length as episode.
    """
    returns = [0.0] * len(episode)
    G = 0.0
    for t in reversed(range(len(episode))):
        _, _, r = episode[t]
        G = r + gamma * G
        returns[t] = G
    return returns

