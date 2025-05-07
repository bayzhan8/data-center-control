from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from .base_agent import BaseAgent
from environment import DataCenterEnv
from config import DataCenterConfig

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent(BaseAgent):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        super().__init__(env, config)
        self.state_size = 4  # (a_t, b_t, q_s, q_l)
        
        # Create action map first
        self.action_map = self._create_action_map()
        self.action_size = len(self.action_map)  # Update action size based on map
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer with DQN learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.dqn_learning_rate)
        
        # Initialize memory buffer with DQN memory size
        self.memory = deque(maxlen=self.config.dqn_memory_size)
        
        # Initialize exploration parameters
        self.epsilon = self.config.dqn_epsilon
    
    def _create_action_map(self) -> List[Tuple[int, int, int, int, int]]:
        """Create a mapping from action indices to actual actions."""
        # Get all possible actions for all possible states
        all_actions = set()
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        actions = self.env.get_possible_actions(state)
                        all_actions.update(actions)
        return sorted(list(all_actions))  # Sort for deterministic ordering
    
    def _state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """Convert state tuple to tensor."""
        return torch.FloatTensor(state).to(self.device)
    
    def _get_action_index(self, action: Tuple[int, int, int, int, int]) -> int:
        """Get index of action in action_map."""
        return self.action_map.index(action)
    
    def _get_action_from_index(self, index: int) -> Tuple[int, int, int, int, int]:
        """Get action from index in action_map."""
        return self.action_map[index]
    
    def _get_valid_actions_mask(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """Create a mask for valid actions in the current state."""
        valid_actions = self.env.get_possible_actions(state)
        mask = torch.zeros(self.action_size, device=self.device)
        for action in valid_actions:
            try:
                idx = self._get_action_index(action)
                if idx < self.action_size:  # Double check index is valid
                    mask[idx] = 1
            except (ValueError, IndexError):
                # If action not in map or index out of bounds, skip it
                continue
        return mask
    
    def _add_to_memory(self, state: Tuple[int, int, int, int], 
                      action: Tuple[int, int, int, int, int],
                      reward: float,
                      next_state: Tuple[int, int, int, int],
                      done: bool) -> None:
        """Add a transition to the replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self) -> Dict[str, Any]:
        """Train the DQN agent."""
        episode_rewards = []
        best_avg_reward = float('-inf')
        no_improvement_count = 0
        
        # Progress bar
        pbar = tqdm(range(self.config.num_episodes), desc="Training DQN agent")
        
        for episode in pbar:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.config.max_steps:
                # Select action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Store transition in memory
                self._add_to_memory(state, action, reward, next_state, done)
                
                # Train on batch if enough samples
                if len(self.memory) >= self.config.dqn_batch_size:
                    self._train_step()
                
                # Move to next state
                state = next_state
                episode_reward += reward
                step += 1
            
            # Update target network periodically
            if episode % self.config.dqn_target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.config.dqn_epsilon_min, 
                             self.epsilon * self.config.dqn_epsilon_decay)
            
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
    
    def _train_step(self):
        """Perform one training step."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.config.dqn_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get action indices
        action_indices = torch.LongTensor([self._get_action_index(a) for a in actions]).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, action_indices.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: choose random valid action
            return random.choice(self.env.get_possible_actions(state))
        
        # Exploitation: choose best valid action
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            
            # Mask invalid actions
            valid_actions_mask = self._get_valid_actions_mask(state)
            q_values = q_values.masked_fill(valid_actions_mask == 0, float('-inf'))
            
            # Get best action
            action_index = q_values.argmax().item()
            return self._get_action_from_index(action_index)
    
    def save(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f"{path}_dqn.pt")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(f"{path}_dqn.pt")
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 