from typing import Tuple, List, Dict, Any
import numpy as np
from config import DataCenterConfig

class DataCenterEnv:
    def __init__(self, config: DataCenterConfig):
        self.config = config
        self.reset()
    
    def reset(self) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
        """Reset the environment to initial state."""
        # Initial state: (a_t, b_t, q^S_t, q^L_t)
        self.state = (0, 0, 0, 0)  # Start with no active servers and empty queues
        return self.state, {}
    
    def get_possible_actions(self, state: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int, int]]:
        """Get all possible actions for the current state."""
        a_t, b_t, q_s, q_l = state
        possible_actions = []
        
        # For each possible number of active servers
        for a_next in range(self.config.N_A + 1):
            for b_next in range(self.config.N_B + 1):
                # For each possible job allocation
                for x_s_a in range(min(a_t, q_s) + 1):
                    remaining_small = q_s - x_s_a
                    for x_s_b in range(min(b_t, remaining_small) + 1):
                        for x_l_b in range(min(b_t - x_s_b, q_l) + 1):
                            possible_actions.append((a_next, b_next, x_s_a, x_s_b, x_l_b))
        
        return possible_actions
    

    def get_transition_probabilities(self, state: Tuple[int, int, int, int], 
                                   action: Tuple[int, int, int, int, int]) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Get all possible next states and their probabilities."""
        a_t, b_t, q_s, q_l = state
        a_next, b_next, x_s_a, x_s_b, x_l_b = action
        
        # 1. Process current job assignments
        completed_small = x_s_a + x_s_b  # R^S_t
        completed_large = x_l_b  # R^L_t
        
        # Update queues after processing
        q_s_after_processing = q_s - completed_small
        q_l_after_processing = q_l - completed_large
        
        # 2. Calculate next states for each possible arrival
        next_states = []
        
        # Case 1: Small job arrives (λ^S_t = 1)
        if q_s_after_processing < self.config.max_queue_small:
            # Can accept the new small job
            q_s_next = q_s_after_processing + 1
            q_l_next = q_l_after_processing
            next_states.append(((a_next, b_next, q_s_next, q_l_next), self.config.p_S))
        else:
            # Small job is dropped
            q_s_next = q_s_after_processing
            q_l_next = q_l_after_processing
            next_states.append(((a_next, b_next, q_s_next, q_l_next), self.config.p_S))
        
        # Case 2: Large job arrives (λ^L_t = 1)
        if q_l_after_processing < self.config.max_queue_large:
            # Can accept the new large job
            q_s_next = q_s_after_processing
            q_l_next = q_l_after_processing + 1
            next_states.append(((a_next, b_next, q_s_next, q_l_next), self.config.p_L))
        else:
            # Large job is dropped
            q_s_next = q_s_after_processing
            q_l_next = q_l_after_processing
            next_states.append(((a_next, b_next, q_s_next, q_l_next), self.config.p_L))
        
        # Case 3: No job arrives
        q_s_next = q_s_after_processing
        q_l_next = q_l_after_processing
        next_states.append(((a_next, b_next, q_s_next, q_l_next), 1 - self.config.p_S - self.config.p_L))
        
        return next_states
    
    def step(self, action: Tuple[int, int, int, int, int]) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        a_next, b_next, x_s_a, x_s_b, x_l_b = action
        a_t, b_t, q_s, q_l = self.state
        
        # 1. Process current job assignments
        completed_small = x_s_a + x_s_b  # R^S_t
        completed_large = x_l_b  # R^L_t
        
        # Update queues after processing
        q_s_after_processing = q_s - completed_small
        q_l_after_processing = q_l - completed_large
        
        # Check for negative queue lengths after processing
        if q_s_after_processing < 0 or q_l_after_processing < 0:
            raise ValueError(f"Invalid negative queue length after processing: small={q_s_after_processing}, large={q_l_after_processing}")
        
        # 2. Apply random arrivals
        rand = np.random.random()
        dropped_jobs = 0
        lambda_s = 0
        lambda_l = 0
        
        if rand < self.config.p_S:
            # Small job arrives (λ^S_t = 1)
            lambda_s = 1
            if q_s_after_processing < self.config.max_queue_small:
                q_s_next = q_s_after_processing + 1
            else:
                q_s_next = q_s_after_processing
                dropped_jobs = 1
            q_l_next = q_l_after_processing
        elif rand < self.config.p_S + self.config.p_L:
            # Large job arrives (λ^L_t = 1)
            lambda_l = 1
            if q_l_after_processing < self.config.max_queue_large:
                q_l_next = q_l_after_processing + 1
            else:
                q_l_next = q_l_after_processing
                dropped_jobs = 1
            q_s_next = q_s_after_processing
        else:
            # No job arrives
            q_s_next = q_s_after_processing
            q_l_next = q_l_after_processing
        
        # 3. Calculate costs and rewards
        # Energy costs
        energy_cost = (self.config.c_A * a_next + 
                      self.config.c_B * b_next)
        
        # Switching costs
        switch_cost = (self.config.c_switch * 
                      (abs(a_next - a_t) + abs(b_next - b_t)))
        
        # Queue costs
        queue_cost = self.config.c_queue * (q_s + q_l)
        
        # Drop costs
        drop_cost = self.config.c_drop * dropped_jobs
        
        # Completion rewards
        completion_reward = (self.config.c_complete_small * completed_small +
                           self.config.c_complete_large * completed_large)
        
        # Total reward (negative cost)
        total_reward = completion_reward - (energy_cost + switch_cost + queue_cost + drop_cost)
        
        # 4. Update state
        self.state = (a_next, b_next, q_s_next, q_l_next)
        
        # Check if episode is done (optional - could be based on time horizon)
        done = False
        
        return self.state, total_reward, done, False, {
            'dropped_jobs': dropped_jobs,
            'completed_small': completed_small,
            'completed_large': completed_large,
            'lambda_s': lambda_s,
            'lambda_l': lambda_l,
            'energy_cost': energy_cost,
            'switch_cost': switch_cost,
            'queue_cost': queue_cost,
            'drop_cost': drop_cost,
            'completion_reward': completion_reward
        } 