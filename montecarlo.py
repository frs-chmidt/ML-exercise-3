from __future__ import annotations

from typing import Dict, Tuple, List, Set, Optional
import numpy as np

from grid import env, state, action
from episode import Transition, compute_returns

"""
montecarlo.py

Implements First-Visit Monte Carlo (MC) Control for learning an action-value
function Q(s, a) in the given grid environment.

Key ideas:
- We estimate Q(s,a) via Monte Carlo returns (complete episode outcomes).
- We use FIRST-VISIT MC: update each (s,a) only the first time it appears
  within a single episode.
- To improve the policy while learning, we generate episodes using an
  epsilon-greedy policy derived from Q.

Data structures:
- state: (row, col, vx, vy)
- action: (ax, ay)
- Q-table: dictionary mapping (state, action) -> estimated expected return
"""

# Convenience alias: state-action key for dictionaries
SA = Tuple[state, action]


def first_visit_mc_update(
    episode: List[Transition],
    q_table: Dict[SA, float],
    visit_counts: Dict[SA, int],
    gamma: float = 1.0,
) -> None:
    """
    Perform a FIRST-VISIT Monte Carlo update on Q(s,a) from a single episode.

    Parameters
    ----------
    episode:
        List of transitions [(s_t, a_t, r_{t+1}), ...] collected until terminal
        or until max_steps.
    q_table:
        Dictionary Q[(s,a)] = estimated expected return starting from (s,a).
    visit_counts:
        Dictionary N[(s,a)] = how many first-visits we have updated for (s,a).
    gamma:
        Discount factor.

    Update rule (incremental mean):
        N(s,a) <- N(s,a) + 1
        Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a)

    Notes
    -----
    - "First-visit" means: within one episode, update a (s,a) at its first
      occurrence only (ignore later occurrences in same episode).
    """
    # G_t for each time step t (computed once, then indexed)
    returns = compute_returns(episode, gamma=gamma)

    seen_pairs: Set[SA] = set()  # to enforce first-visit updates

    for t, (s, a, _r) in enumerate(episode):
        sa = (s, a)

        # Skip if we've already updated this (s,a) in the current episode
        if sa in seen_pairs:
            continue
        seen_pairs.add(sa)

        G = returns[t]

        # Update visit count
        visit_counts[sa] = visit_counts.get(sa, 0) + 1
        n = visit_counts[sa]

        # Incremental mean update for Q
        old_q = q_table.get(sa, 0.0)  # unseen pairs start at 0
        q_table[sa] = old_q + (G - old_q) / n


def eps_greedy_action(
    e: env,
    q_table: Dict[SA, float],
    s: state,
    epsilon: float,
) -> action:
    """
    Choose an action using an epsilon-greedy policy derived from Q.

    With probability epsilon:
        pick a random action (exploration)
    With probability (1 - epsilon):
        pick the action with the largest Q(s,a) (exploitation)

    Unseen (s,a) pairs are treated as Q(s,a) = 0.0.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1].")

    # Exploration: random action from the action space
    if np.random.random() < epsilon:
        idx = np.random.randint(len(e.action_space))
        return e.action_space[idx]

    # Exploitation: pick argmax_a Q(s,a)
    best_a: Optional[action] = None
    best_q = -float("inf")

    for a in e.action_space:
        q_val = q_table.get((s, a), 0.0)
        if q_val > best_q:
            best_q = q_val
            best_a = a

    # Fallback (should not happen if action_space is non-empty)
    if best_a is None:
        idx = np.random.randint(len(e.action_space))
        best_a = e.action_space[idx]

    return best_a


def generate_episode_eps_greedy(
    e: env,
    q_table: Dict[SA, float],
    epsilon: float,
    max_steps: int = 500,
) -> List[Transition]:
    """
    Generate ONE episode by interacting with the environment using the current
    epsilon-greedy policy.

    Returns
    -------
    episode:
        List of (state, action, reward) transitions.
        Note: We do not store next_state in the episode because MC updates only
        need the sequence of rewards and the (s,a) pairs.
    """
    episode: List[Transition] = []
    s = e.reset()

    for _ in range(max_steps):
        a = eps_greedy_action(e, q_table, s, epsilon)
        s_next, r, done, _info = e.step(a)

        episode.append((s, a, r))
        s = s_next

        if done:
            break

    return episode


def mc_control_train(
    e: env,
    num_episodes: int = 5000,
    epsilon: float = 0.2,
    gamma: float = 1.0,
    max_steps: int = 500,
    seed: int = 0,
) -> Tuple[Dict[SA, float], Dict[SA, int]]:
    """
    Train a Monte Carlo Control agent (First-Visit MC) using epsilon-greedy
    improvement.

    Parameters
    ----------
    e:
        Environment instance (grid.env).
    num_episodes:
        Number of episodes to sample for learning.
    epsilon:
        Starting epsilon (exploration rate). We linearly decay it.
    gamma:
        Discount factor for returns.
    max_steps:
        Maximum steps per episode before truncation.
    seed:
        Random seed for reproducibility (affects epsilon-greedy exploration).

    Returns
    -------
    q_table:
        Learned action-value function Q(s,a).
    visit_counts:
        First-visit update counts N(s,a) (useful for debugging/analysis).

    Notes
    -----
    We use a simple linear epsilon decay:
        eps(ep) goes from epsilon_start -> epsilon_end over num_episodes.
    """
    np.random.seed(seed)

    q_table: Dict[SA, float] = {}
    visit_counts: Dict[SA, int] = {}

    epsilon_start = epsilon
    epsilon_end = 0.05  # keep a little exploration until the end

    for ep in range(num_episodes):
        # Linear decay of epsilon across training
        frac = ep / max(1, num_episodes - 1)
        eps = epsilon_start + frac * (epsilon_end - epsilon_start)

        # 1) Generate one episode under current epsilon-greedy policy
        episode = generate_episode_eps_greedy(
            e,
            q_table,
            eps,
            max_steps=max_steps,
        )

        # 2) Improve Q using First-Visit Monte Carlo estimates
        first_visit_mc_update(
            episode,
            q_table,
            visit_counts,
            gamma=gamma,
        )

    return q_table, visit_counts
