from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
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
        self.action_size = len(self.env.get_possible_actions((0, 0, 0, 0)))  # Number of possible actions
        
        # DQN specific parameters
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.batch_size = 64
        self.gamma = config.gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Action mapping
        self.action_map = self._create_action_map()
    
    def _create_action_map(self) -> List[Tuple[int, int, int, int, int]]:
        """Create a mapping from action indices to actual actions."""
        return self.env.get_possible_actions((0, 0, 0, 0))
    
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
            mask[self._get_action_index(action)] = 1
        return mask
    
    def train(self) -> Dict[str, Any]:
        """Train the DQN agent."""
        episode_rewards = []
        
        for episode in range(self.config.num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition in memory
                self.memory.append((state, action, reward, next_state, done))
                
                # Move to next state
                state = next_state
                total_reward += reward
                
                # Train on batch if enough samples
                if len(self.memory) >= self.batch_size:
                    self._train_step()
            
            # Update target network periodically
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")
        
        return {
            'episodes_trained': self.config.num_episodes,
            'final_epsilon': self.epsilon,
            'final_avg_reward': np.mean(episode_rewards[-100:])
        }
    
    def _train_step(self):
        """Perform one training step."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
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
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
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
        """Save the model and training state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, f"{path}_dqn.pt")
    
    def load(self, path: str) -> None:
        """Load the model and training state."""
        checkpoint = torch.load(f"{path}_dqn.pt")
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 