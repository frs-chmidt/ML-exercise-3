from __future__ import annotations

from typing import Dict, Tuple, List, Set
import numpy as np

from grid import env, state, action
from episode import Transition, compute_returns


# We store values for "state-action" pairs.
# state = (row, col, vx, vy)
# action = (ax, ay)
SA = Tuple[state, action]


def first_visit_mc_update(
    episode: List[Transition],
    q_table: Dict[SA, float],
    visit_counts: Dict[SA, int],
    gamma: float = 1.0,
) -> None:
    """
    First-Visit Monte Carlo update for Q(s,a).

    episode: list of transitions [(s, a, r), ...]
    q_table[(s,a)]: estimated expected return after taking action a in state s
    visit_counts[(s,a)]: how many (first) visits we have seen for that pair

    Update rule (incremental mean):
        N(s,a) <- N(s,a) + 1
        Q(s,a) <- Q(s,a) + (G - Q(s,a)) / N(s,a)
    """
    returns = compute_returns(episode, gamma=gamma)

    # We only update each (state, action) pair once per episode (first-visit MC)
    seen_pairs: Set[SA] = set()

    for t, (s, a, _r) in enumerate(episode):
        sa = (s, a)
        if sa in seen_pairs:
            continue
        seen_pairs.add(sa)

        G = returns[t]

        # Count this (first) visit
        visit_counts[sa] = visit_counts.get(sa, 0) + 1
        n = visit_counts[sa]

        # Update Q with incremental mean
        old_q = q_table.get(sa, 0.0)
        q_table[sa] = old_q + (G - old_q) / n


def eps_greedy_action(
    e: env,
    q_table: Dict[SA, float],
    s: state,
    epsilon: float,
) -> action:
    """
    Epsilon-greedy policy derived from Q:

    - With probability epsilon: explore (random action)
    - With probability 1-epsilon: exploit (choose action with highest Q)

    Note: If a (state, action) was never seen, we treat Q(s,a)=0.0.
    """
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    # Exploration: random action
    if np.random.random() < epsilon:
        return e.action_space[np.random.randint(len(e.action_space))]

    # Exploitation: choose argmax_a Q(s,a)
    best_action: action | None = None
    best_value = -float("inf")

    for a in e.action_space:
        value = q_table.get((s, a), 0.0)
        if value > best_value:
            best_value = value
            best_action = a

    # Safety fallback (should not happen)
    if best_action is None:
        best_action = e.action_space[np.random.randint(len(e.action_space))]

    return best_action


def generate_episode_eps_greedy(
    e: env,
    q_table: Dict[SA, float],
    epsilon: float,
    max_steps: int = 500,
) -> List[Transition]:
    """
    Generate ONE episode using the current epsilon-greedy policy.
    We record (state, action, reward) for each step.
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
    seed: int = 0
) -> Tuple[Dict[SA, float], Dict[SA, int]]:

    np.random.seed(seed)

    q_table: Dict[SA, float] = {}
    visit_counts: Dict[SA, int] = {}

    # ↓↓↓ EVERYTHING BELOW MUST BE INDENTED ↓↓↓
    epsilon_start = epsilon
    epsilon_end = 0.05

    for ep in range(num_episodes):
        frac = ep / max(1, num_episodes - 1)
        eps = epsilon_start + frac * (epsilon_end - epsilon_start)

        episode = generate_episode_eps_greedy(
            e,
            q_table,
            eps,
            max_steps=max_steps
        )

        first_visit_mc_update(
            episode,
            q_table,
            visit_counts,
            gamma=gamma
        )

    return q_table, visit_counts


