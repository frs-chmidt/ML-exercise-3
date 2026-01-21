from typing import List, Tuple
from grid import env, state, action
from montecarlo import eps_greedy_action

def greedy_path(
    e: env,
    Q: dict,
    max_steps: int = 200
) -> List[state]:
    """
    Run ONE episode using the greedy policy (epsilon = 0)
    and return the visited states.
    """
    path: List[state] = []

    s = e.reset()
    path.append(s)

    for _ in range(max_steps):
        a = eps_greedy_action(e, Q, s, epsilon=0.0)
        s, r, done, info = e.step(a)
        path.append(s)

        if done:
            break

    return path
