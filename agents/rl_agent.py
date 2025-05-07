from typing import Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from .base_agent import BaseAgent
from environment import DataCenterEnv
from config import DataCenterConfig

class QLearningAgent(BaseAgent):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        super().__init__(env, config)
        self.q_table = {}  # Maps (state, action) pairs to Q-values
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate
        self._initialize_q_table()
    
    def _initialize_q_table(self):
        """Initialize Q-table for all possible state-action pairs."""
        # Initialize Q-values to 0 (we want to maximize rewards)
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        for action in self.env.get_possible_actions(state):
                            self.q_table[(state, action)] = 0.0
    
    def _get_q_value(self, state: Tuple[int, int, int, int], 
                    action: Tuple[int, int, int, int, int]) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get((state, action), 0.0)
    
    def _get_best_action(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Get best action for state based on Q-values."""
        possible_actions = self.env.get_possible_actions(state)
        if not possible_actions:
            return None
        
        # Get Q-values for all possible actions
        q_values = [self._get_q_value(state, action) for action in possible_actions]
        best_idx = np.argmax(q_values)  # MAXIMIZING
        return possible_actions[best_idx]
    
    def train(self) -> Dict[str, Any]:
        """Train the agent using vanilla Q-learning."""
        episode_rewards = []
        best_avg_reward = float('-inf')
        no_improvement_count = 0
        
        # Progress bar
        pbar = tqdm(range(self.config.num_episodes), desc="Training RL agent")
        
        for episode in pbar:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.config.max_steps:
                # Epsilon-greedy action selection
                if np.random.random() < self.epsilon:
                    # Explore: choose random action
                    possible_actions = self.env.get_possible_actions(state)
                    action = possible_actions[np.random.randint(len(possible_actions))]
                else:
                    # Exploit: choose best action
                    action = self._get_best_action(state)
                
                # Get next state and reward using cached transitions
                transitions = self.transition_cache[(state, action)]
                next_states = []
                probs = []
                rewards = []
                
                for next_state, prob, reward in transitions:
                    next_states.append(next_state)
                    probs.append(prob)
                    rewards.append(reward)
                
                # Sample next state based on probabilities
                next_state_idx = np.random.choice(len(next_states), p=probs)
                next_state = next_states[next_state_idx]
                reward = rewards[next_state_idx]
                
                # Q-learning update
                old_q = self._get_q_value(state, action)
                next_best_action = self._get_best_action(next_state)
                next_q = self._get_q_value(next_state, next_best_action)
                
                # Update Q-value (maximizing)
                new_q = old_q + self.config.learning_rate * (
                    reward + self.config.gamma * next_q - old_q
                )
                self.q_table[(state, action)] = new_q
                
                episode_reward += reward
                state = next_state
                step += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            episode_rewards.append(episode_reward)
            
            # Update progress bar
            avg_reward = np.mean(episode_rewards[-100:])
            pbar.set_postfix({
                'avg_reward': f'{avg_reward:.2f}',
                'epsilon': f'{self.epsilon:.3f}'
            })
            
            # Early stopping check
            if len(episode_rewards) >= self.config.min_episodes:
                current_avg = np.mean(episode_rewards[-100:])
                if current_avg > best_avg_reward:  # MAXIMIZING
                    best_avg_reward = current_avg
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.config.max_no_improvement:
                    print(f"\nEarly stopping at episode {episode + 1}")
                    break
        
        return {
            'episode_rewards': episode_rewards,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'episodes_trained': len(episode_rewards),
            'final_epsilon': self.epsilon
        }
    
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            possible_actions = self.env.get_possible_actions(state)
            return possible_actions[np.random.randint(len(possible_actions))]
        else:
            # Exploit: choose best action
            return self._get_best_action(state)
    
    def save(self, path: str) -> None:
        """Save Q-table and epsilon."""
        np.save(f"{path}_q_table.npy", self.q_table)
        np.save(f"{path}_epsilon.npy", self.epsilon)
    
    def load(self, path: str) -> None:
        """Load Q-table and epsilon."""
        self.q_table = np.load(f"{path}_q_table.npy", allow_pickle=True).item()
        self.epsilon = np.load(f"{path}_epsilon.npy") 