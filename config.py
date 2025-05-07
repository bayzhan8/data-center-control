from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataCenterConfig:
    # Server configuration
    N_A: int = 2  # Total number of Type A servers
    N_B: int = 3  # Total number of Type B servers
    
    # Queue configuration
    max_queue_small: int = 3  # Maximum queue length for small jobs
    max_queue_large: int = 2  # Maximum queue length for large jobs
    
    # Cost parameters
    c_A: float = 1.0  # Energy cost per active A server
    c_B: float = 2.0  # Energy cost per active B server
    c_switch: float = 0.5  # Cost of changing server state
    c_queue: float = 0.1  # Cost per job waiting in queue
    c_drop: float = 5.0  # Cost of dropping a job
    
    # Completion rewards
    c_complete_small: float = 2.0  # Reward for completing a small job
    c_complete_large: float = 4.0  # Reward for completing a large job
    
    # Job arrival probabilities
    p_S: float = 0.3  # Probability of small job arrival
    p_L: float = 0.2  # Probability of large job arrival
    
    # Algorithm parameters
    gamma: float = 0.99  # Discount factor
    epsilon: float = 0.1  # Exploration rate for RL
    learning_rate: float = 0.1  # Learning rate for RL
    max_iterations: int = 1000  # Maximum iterations for VI/PI
    max_steps: int = 1000  # Maximum steps for RL
    convergence_threshold: float = 1e-6  # Convergence threshold for VI/PI
    
    # DQN specific parameters
    dqn_epsilon: float = 1.0  # Start with full exploration
    dqn_epsilon_min: float = 0.05  # Increased minimum exploration rate
    dqn_epsilon_decay: float = 0.99  # Faster decay
    dqn_learning_rate: float = 0.001  # Learning rate for DQN
    dqn_batch_size: int = 32  # Smaller batch size for faster training
    dqn_target_update: int = 5  # More frequent target updates
    dqn_memory_size: int = 5000  # Smaller memory size
    
    # Training parameters
    num_episodes: int = 500  # Reduced number of episodes
    eval_episodes: int = 10   # Number of episodes for evaluation
    
    # Early stopping parameters
    min_episodes: int = 200  # Reduced minimum episodes
    max_no_improvement: int = 50  # Reduced patience for early stopping
    
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
            'c_complete_small': self.c_complete_small,
            'c_complete_large': self.c_complete_large,
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
            'eval_episodes': self.eval_episodes,
            'min_episodes': self.min_episodes,
            'max_no_improvement': self.max_no_improvement
        }

# Default configuration
default_config = DataCenterConfig() 