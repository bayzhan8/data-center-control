from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataCenterConfig:
    # Server configuration
    N_A: int = 2  # Total number of Type A servers
    N_B: int = 3  # Total number of Type B servers
    
    # Cost parameters
    c_A: float = 1.0  # Energy cost per active A server
    c_B: float = 2.0  # Energy cost per active B server
    c_switch: float = 0.5  # Cost of changing server state
    c_queue: float = 0.2  # Cost per job waiting in queue
    c_drop: float = 5.0  # Penalty cost per dropped job
    
    # Job arrival probabilities
    p_S: float = 0.4  # Probability of small job arrival
    p_L: float = 0.3  # Probability of large job arrival
    
    # Queue limits
    max_queue_small: int = 2  # Maximum number of small jobs in queue
    max_queue_large: int = 3  # Maximum number of large jobs in queue
    
    # Algorithm parameters
    gamma: float = 0.95  # Discount factor
    epsilon: float = 0.1  # Exploration rate for RL
    learning_rate: float = 0.1  # Learning rate for RL
    max_iterations: int = 1000  # Maximum iterations for VI/PI
    convergence_threshold: float = 1e-6  # Convergence threshold for VI/PI
    
    # Training parameters
    num_episodes: int = 2000  # Number of episodes for RL training
    eval_episodes: int = 1000  # Increased from 100 to 1000 for better statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for easier serialization."""
        return {
            'N_A': self.N_A,
            'N_B': self.N_B,
            'c_A': self.c_A,
            'c_B': self.c_B,
            'c_switch': self.c_switch,
            'c_queue': self.c_queue,
            'c_drop': self.c_drop,
            'p_S': self.p_S,
            'p_L': self.p_L,
            'max_queue_small': self.max_queue_small,
            'max_queue_large': self.max_queue_large,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'num_episodes': self.num_episodes,
            'eval_episodes': self.eval_episodes
        }

# Default configuration
default_config = DataCenterConfig() 