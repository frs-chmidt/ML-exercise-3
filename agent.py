''' here we will define the agent that will interact with the environment.
In other words we will define all possible actions the agent can take and the state it is in.
'''

#imports
from __future__ import annotations #for forward references
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
        if not (0.0 <= epsilon <= 1.0): # epsilon defines exploration rate
            raise ValueError("epsilon must be in [0, 1].")
        if not (0.0 <= gamma <= 1.0): #gamma degfines discount factor
            raise ValueError("gamma must be in [0, 1].")

        #store parameters
        self.action_space: List[action] = list(action_space)
        self.epsilon = float(epsilon)  #exploration rate
        self.gamma = float(gamma)      #discount factor
        self.rng = np.random.default_rng(seed)
        
        #create initially empty Q and N tables
        self.Q: Dict[Tuple[state, action], float] = {} #If I take action a in state s and then follow my policy afterwards, what return do I expect? (=estimated value)
        self.N: Dict[Tuple[state, action], int] = {} # stores how many times I have taken action a in state s this is used to compute average returns

    #if (s,a) hasn’t been seen, treat its value as 0.0.
    def q(self, s: state, a: action) -> float:
        return self.Q.get((s, a), 0.0)
    
    #Ensures that every visited (s,a) has both a Q-value and a visit count.
    def ensure_entry(self, s: state, a: action):
        key = (s, a)
        if key not in self.Q:
            self.Q[key] = 0.0
            self.N[key] = 0
    
    #computes Q values for all actions in state s. finds the best action(s). returns list of best actions (ties are okay)
    def greedy_actions(self, s: state) -> List[action]:
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

    #returns a dictionary mapping each action to its probability under the current policy
    def policy_probs(self, s: state) -> Dict[action, float]:
        n_actions = len(self.action_space)
        greedy = self.greedy_actions(s)
        k = len(greedy)

        probs: Dict[action, float] = {a: self.epsilon / n_actions for a in self.action_space}
        bonus = (1.0 - self.epsilon) / k
        for a in greedy:
            probs[a] += bonus
        return probs

    #greedy=True (evaluation mode) (Compute all greedy actions g and choose uniformly among them). greedy=False (training mode) (compute epsilon-greedy distribution and sample from it using rng.choice).
    def select_action(self, s: state, greedy: bool = False) -> action:
        if greedy == True:
            g = self.greedy_actions(s)
            return g[int(self.rng.integers(0, len(g)))]
            
        probs = self.policy_probs(s)
        actions = self.action_space
        p = np.array([probs[a] for a in actions], dtype=float)
        idx = self.rng.choice(len(actions), p=p)
        return actions[int(idx)]
    
    def update_from_episode(self, episode: List[Tuple[state, action, float]])->None:
        
        if not episode:
            return #If the episode list is empty, nothing to update
        
        G = 0.0
        returns = [0.0] * len(episode)

        for t in range(len(episode) - 1, -1, -1): 
            s, a, r = episode[t]
            G = r + self.gamma * G
            returns[t] = G #compute and store the return for each time step t
        
        #We use first-visit Monte Carlo updates (each state-action pair updated once per episode) for simplicity
            seen = set()
            for t in range (len(episode)):
                s, a, r = episode[t]
                key = (s, a)
                if key in seen:
                    continue
                seen.add(key)
                self.update_q(s, a, returns[t])
    
    '''Each time I see (state s, action a), I look at how good the rest of the episode turned out.
    I then update my estimate of how good (s,a) is by averaging this outcome with all previous outcomes.
    update_q stores the average return for each (state, action) pair across all visits
    Not immediate reward, not per time-step index — but full episode return starting from that state-action.
    '''
    def update_q(self, s: state, a: action, G: float)-> None:
        self.ensure_entry(s, a)
        key = (s, a)
        self.N[key] += 1
        n = self.N[key]
        self.Q[key] +=  (G - self.Q[key]) / n
        
        
    