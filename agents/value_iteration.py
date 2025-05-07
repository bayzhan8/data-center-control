from typing import Tuple, Dict, Any
import numpy as np
from .base_agent import BaseAgent
from environment import DataCenterEnv
from config import DataCenterConfig

class ValueIterationAgent(BaseAgent):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        super().__init__(env, config)
        self.value_function = {}  # Maps states to their values
        self.policy = {}  # Maps states to their optimal actions
        self.transition_cache = {}  # Cache for transition probabilities and rewards
        self._initialize_value_function()
        self._initialize_transition_cache()
    
    def _initialize_value_function(self):
        """Initialize value function for all possible states."""
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        self.value_function[state] = 0.0
    
    def _initialize_transition_cache(self):
        """Initialize cache for transition probabilities and rewards."""
        for state in self.value_function.keys():
            for action in self.env.get_possible_actions(state):
                transitions = []
                for next_state, prob in self.env.get_transition_probabilities(state, action):
                    # Calculate reward directly without environment step
                    a_next, b_next, x_s_a, x_s_b, x_l_b = action
                    a_t, b_t, q_s, q_l = state
                    
                    # Calculate costs
                    energy_cost = (self.config.c_A * a_next + 
                                 self.config.c_B * b_next)
                    switch_cost = (self.config.c_switch * 
                                 (abs(a_next - a_t) + abs(b_next - b_t)))
                    queue_cost = self.config.c_queue * (q_s + q_l)
                    
                    # Calculate dropped jobs
                    dropped_jobs = 0
                    if q_s == self.config.max_queue_small and prob == self.config.p_S:
                        dropped_jobs = 1
                    elif q_l == self.config.max_queue_large and prob == self.config.p_L:
                        dropped_jobs = 1
                    
                    # Calculate drop cost
                    drop_cost = self.config.c_drop * dropped_jobs
                    
                    # Total cost (negative reward)
                    total_cost = energy_cost + switch_cost + queue_cost + drop_cost
                    reward = -total_cost
                    
                    transitions.append((next_state, prob, reward))
                
                self.transition_cache[(state, action)] = transitions
    
    def train(self) -> Dict[str, Any]:
        """Run value iteration to find optimal policy."""
        iteration = 0
        max_diff = float('inf')
        
        while iteration < self.config.max_iterations and max_diff > self.config.convergence_threshold:
            max_diff = 0.0
            
            # Update value function for each state
            for state in self.value_function.keys():
                old_value = self.value_function[state]
                new_value, best_action = self._compute_value(state)
                
                # Update value function and policy
                self.value_function[state] = new_value
                self.policy[state] = best_action
                
                # Track maximum change
                max_diff = max(max_diff, abs(new_value - old_value))
            
            iteration += 1
        
        return {
            'iterations': iteration,
            'final_max_diff': max_diff
        }
    
    def _compute_value(self, state: Tuple[int, int, int, int]) -> Tuple[float, Tuple[int, int, int, int, int]]:
        """Compute value for a state using Bellman equation."""
        best_value = float('-inf')
        best_action = None
        
        # Try all possible actions
        for action in self.env.get_possible_actions(state):
            action_value = 0.0
            transitions = self.transition_cache[(state, action)]
            
            # Compute expected value over all possible next states
            for next_state, prob, reward in transitions:
                action_value += prob * (reward + self.config.gamma * self.value_function[next_state])
            
            # Update best value and action
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_value, best_action
    
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select action using the learned policy."""
        if state not in self.policy:
            # If state not in policy (shouldn't happen), use random action
            return self.env.get_possible_actions(state)[0]
        return self.policy[state]
    
    def save(self, path: str) -> None:
        """Save value function and policy."""
        np.save(f"{path}_value_function.npy", self.value_function)
        np.save(f"{path}_policy.npy", self.policy)
    
    def load(self, path: str) -> None:
        """Load value function and policy."""
        self.value_function = np.load(f"{path}_value_function.npy", allow_pickle=True).item()
        self.policy = np.load(f"{path}_policy.npy", allow_pickle=True).item() 