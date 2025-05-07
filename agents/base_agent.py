from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from environment import DataCenterEnv
from config import DataCenterConfig

class BaseAgent(ABC):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        self.env = env
        self.config = config
        self.transition_cache = {}  # Cache for transition probabilities and rewards
        self._initialize_transition_cache()
    
    def _initialize_transition_cache(self):
        """Initialize cache for transition probabilities and rewards."""
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
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
                                
                                # Calculate completion rewards
                                completed_small = x_s_a + x_s_b
                                completed_large = x_l_b
                                completion_reward = (self.config.c_complete_small * completed_small +
                                                   self.config.c_complete_large * completed_large)
                                
                                # Total reward (completion reward minus costs)
                                total_reward = completion_reward - (energy_cost + switch_cost + queue_cost + drop_cost)
                                
                                transitions.append((next_state, prob, total_reward))
                            
                            self.transition_cache[(state, action)] = transitions
    
    @abstractmethod
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select an action for the given state."""
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the agent (if applicable)."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's state."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's state."""
        pass 