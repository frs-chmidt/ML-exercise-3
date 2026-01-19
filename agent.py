''' here we will define the agent that will interact with the environment.
In other words we will define all possible actions the agent can take and the state it is in.
'''

#imports
from __future__ import annotations
from typing import Tuple, List, Any, Dict
import numpy as np
from grid import env, state, action



#action (move in x and y direction by ax, ay)
action = Tuple[int, int]

#state defined by (row, col, vx, vy)
state = Tuple[int, int, int, int]

#define agent class=============================================================
class agent:
    def __init__(self, action_space: List[action], epsilon: float = 0.1, gamma: float = 0.9, seed: int = 36):
        #some assertions
        if not action_space:
            raise ValueError("action_space must be a non-empty list of actions.")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1].")

        self.action_space: List[action] = list(action_space)
        self.epsilon = float(epsilon)  #exploration rate
        self.gamma = float(gamma)      #discount factor
        self.rng = np.random.default_rng(seed)
        
        #create initially empty Q and N tables
        self.Q: Dict[Tuple[State, Action], float] = {}
        self.N: Dict[Tuple[State, Action], int] = {}
        
        def q(self, s: state, a: action) -> float:
            return self.Q.get((s, a), 0.0)
        
        def ensure_entry(self, s: state, a: action):
            key = (s, a)
            if key not in self.Q:
                self.Q[key] = 0.0
                self.N[key] = 0
        
        def greedy_actions(self, s: State) -> List[Action]:
            best = None
            greedy: List[action] = []

            for a in self.action_space:
                v = self.q(s, a)
                if best is None or v > best:
                    best = v
                    greedy = [a]
                elif v == best:
                    greedy.append(a)
            return greedy
        
        def policy_probs(self, s: State) -> Dict[Action, float]:
            n_actions = len(self.action_space)
            greedy = self.greedy_actions(s)
            k = len(greedy)

            probs: Dict[Action, float] = {a: self.epsilon / n_actions for a in self.action_space}
            bonus = (1.0 - self.epsilon) / k
            for a in greedy:
                probs[a] += bonus
            return probs
        
        def select_action(self, s: state, greedy: bool = False) -> action:
            if greedy == True:
                g = self.greedy_actions(s)
                return g[int(self.rng.integers(0, len(g)))]
            
            probs = self.policy_probs(s)
            actions = self.action_space
            p = np.array([probs[a] for a in actions], dtype=float)
            idx = self.rng.choice(len(actions), p=p)
            return actions[int(idx)]
        
        def update_from_episode(self, episode: List[Tuple[state, action, float]], first_visit: bool = True)->None:
            
            if not episode:
                return
            
            G = 0.0
            returns = [0.0] * len(episode)

            for t in range(len(episode) - 1, -1, -1): 
                s, a, r = episode[t]
                G = r + self.gamma * G
                returns[t] = G
                
            if first_visit == True:
                seen = set()
                for t in range (len(episode)):
                    s, a, r = episode[t]
                    key = (s, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    self.update_q(s, a, returns[t])
            else:
                for t in range(len(episode)):
                    s, a, r = episode[t]
                    self.update_q(s, a, returns[t])
                    
        def update_q(self, s: state, a: action, G: float)-> None:
            self.ensure_entry(s, a)
            key = (s, a)
            self.N[key] += 1
            n = self.N[key]
            self.Q[key] +=  (G - self.Q[key]) / n
    