from typing import Tuple, Dict, Any
import numpy as np
from .base_agent import BaseAgent
from environment import DataCenterEnv
from config import DataCenterConfig

class PolicyIterationAgent(BaseAgent):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        super().__init__(env, config)
        self.value_function = {}  # Maps states to their values
        self.policy = {}  # Maps states to their actions
        self.transition_cache = {}  # Cache for transition probabilities and rewards
        self._initialize_value_function()
        self._initialize_policy()
        self._initialize_transition_cache()
    
    def _initialize_value_function(self):
        """Initialize value function for all possible states."""
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        self.value_function[state] = 0.0
    
    def _initialize_policy(self):
        """Initialize policy with random actions."""
        for state in self.value_function.keys():
            possible_actions = self.env.get_possible_actions(state)
            self.policy[state] = possible_actions[0]  # Start with first possible action
    
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
        """Run policy iteration to find optimal policy."""
        iteration = 0
        policy_stable = False
        
        while not policy_stable and iteration < self.config.max_iterations:
            # Policy evaluation (modified to be more efficient)
            self._modified_policy_evaluation()
            
            # Policy improvement
            policy_stable = self._policy_improvement()
            
            iteration += 1
        
        return {
            'iterations': iteration,
            'policy_stable': policy_stable
        }
    
    def _modified_policy_evaluation(self):
        """More efficient policy evaluation using modified policy iteration."""
        # Use a small number of iterations instead of full convergence
        for _ in range(5):  # Can adjust this number
            for state in self.value_function.keys():
                action = self.policy[state]
                transitions = self.transition_cache[(state, action)]
                
                # Compute value for current policy
                new_value = 0.0
                for next_state, prob, reward in transitions:
                    new_value += prob * (reward + self.config.gamma * self.value_function[next_state])
                
                self.value_function[state] = new_value
    
    def _policy_improvement(self) -> bool:
        """Improve policy based on current value function."""
        policy_stable = True
        
        for state in self.value_function.keys():
            old_action = self.policy[state]
            
            # Find best action for current state
            best_value = float('-inf')
            best_action = None
            
            for action in self.env.get_possible_actions(state):
                action_value = 0.0
                transitions = self.transition_cache[(state, action)]
                
                # Compute value for this action
                for next_state, prob, reward in transitions:
                    action_value += prob * (reward + self.config.gamma * self.value_function[next_state])
                
                # Update best action
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            # Update policy
            self.policy[state] = best_action
            
            # Check if policy has changed
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
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