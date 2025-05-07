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
        self._initialize_value_function()
        self._initialize_policy()
    
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
        """Full policy evaluation until convergence."""
        max_diff = float('inf')
        while max_diff > self.config.convergence_threshold:
            max_diff = 0.0
            for state in self.value_function.keys():
                old_value = self.value_function[state]
                action = self.policy[state]
                transitions = self.transition_cache[(state, action)]
                
                # Compute value for current policy
                new_value = 0.0
                for next_state, prob, reward in transitions:
                    new_value += prob * (reward + self.config.gamma * self.value_function[next_state])
                
                self.value_function[state] = new_value
                max_diff = max(max_diff, abs(new_value - old_value))
    
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