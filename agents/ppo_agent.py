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

class PPONetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(PPONetwork, self).__init__()
        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.shared(x)
        return self.policy(features), self.value(features)

class PPOAgent(BaseAgent):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        super().__init__(env, config)
        self.state_size = 4  # (a_t, b_t, q_s, q_l)
        
        # Create action map and cache
        self.action_map = self._create_action_map()
        self.action_size = len(self.action_map)
        
        # Initialize network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.ppo_learning_rate)
        
        # PPO specific parameters
        self.clip_ratio = self.config.ppo_clip_ratio
        self.value_coef = self.config.ppo_value_coef
        self.entropy_coef = self.config.ppo_entropy_coef
        
        # Pre-compute state tensors
        self.state_tensors = {}
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        self.state_tensors[state] = torch.FloatTensor(state).to(self.device)
    
    def _create_action_map(self) -> List[Tuple[int, int, int, int, int]]:
        """Create a mapping from action indices to actual actions."""
        # Cache all possible actions for each state
        self.action_cache = {}
        all_actions = set()
        
        # Pre-compute all possible states and actions
        for a in range(self.config.N_A + 1):
            for b in range(self.config.N_B + 1):
                for q_s in range(self.config.max_queue_small + 1):
                    for q_l in range(self.config.max_queue_large + 1):
                        state = (a, b, q_s, q_l)
                        actions = self.env.get_possible_actions(state)
                        self.action_cache[state] = actions
                        all_actions.update(actions)
        
        return sorted(list(all_actions))
    
    def _state_to_tensor(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """Use pre-computed tensor."""
        return self.state_tensors[state]
    
    def _get_action_index(self, action: Tuple[int, int, int, int, int]) -> int:
        """Get index of action in action_map."""
        return self.action_map.index(action)
    
    def _get_action_from_index(self, index: int) -> Tuple[int, int, int, int, int]:
        """Get action from index in action_map."""
        return self.action_map[index]
    
    def _get_valid_actions_mask(self, state: Tuple[int, int, int, int]) -> torch.Tensor:
        """Create a mask for valid actions in the current state using cache."""
        valid_actions = self.action_cache[state]
        mask = torch.zeros(self.action_size, device=self.device)
        for action in valid_actions:
            try:
                idx = self._get_action_index(action)
                if idx < self.action_size:
                    mask[idx] = 1
            except (ValueError, IndexError):
                continue
        return mask
    
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select action using policy network."""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            action_probs, _ = self.network(state_tensor)
            
            # Mask invalid actions
            valid_actions_mask = self._get_valid_actions_mask(state)
            action_probs = action_probs * valid_actions_mask
            action_probs = action_probs / action_probs.sum()  # Renormalize
            
            # Sample action
            action_index = torch.multinomial(action_probs, 1).item()
            return self._get_action_from_index(action_index)
    
    def train(self) -> Dict[str, Any]:
        """Train the PPO agent."""
        episode_rewards = []
        best_avg_reward = float('-inf')
        no_improvement_count = 0
        
        pbar = tqdm(range(self.config.num_episodes), desc="Training PPO agent")
        
        for episode in pbar:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            # Collect trajectory
            states, actions, rewards, next_states, dones = [], [], [], [], []
            log_probs, values = [], []
            
            while not done and step < self.config.max_steps:
                # Get action probabilities and value
                state_tensor = self._state_to_tensor(state)
                action_probs, value = self.network(state_tensor)
                
                # Mask invalid actions
                valid_actions_mask = self._get_valid_actions_mask(state)
                action_probs = action_probs * valid_actions_mask
                action_probs = action_probs / action_probs.sum()
                
                # Sample action
                action_index = torch.multinomial(action_probs, 1).item()
                action = self._get_action_from_index(action_index)
                
                # Take action
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action_index)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                log_probs.append(torch.log(action_probs[action_index]))
                values.append(value)
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # Compute returns and advantages
            returns = []
            advantages = []
            R = 0
            
            for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
                R = r + self.config.gamma * R * (1 - d)
                returns.insert(0, R)
                advantages.insert(0, R - v.item())
            
            # Convert to tensors
            states = torch.stack([self.state_tensors[s] for s in states])
            actions = torch.LongTensor(actions).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            old_log_probs = torch.stack(log_probs).detach()
            
            # PPO update
            for _ in range(self.config.ppo_epochs):
                # Get new action probabilities and values
                action_probs, values = self.network(states)
                action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
                
                # Compute ratios
                ratios = torch.exp(torch.log(action_probs) - old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(values.squeeze(), returns)
                
                # Compute entropy bonus
                entropy = -(action_probs * torch.log(action_probs)).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            # Update progress
            avg_reward = np.mean(episode_rewards[-100:])
            pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}'})
            
            # Early stopping
            if len(episode_rewards) >= self.config.min_episodes:
                current_avg = np.mean(episode_rewards[-100:])
                if current_avg > best_avg_reward:
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
            'episodes_trained': len(episode_rewards)
        }
    
    def save(self, path: str) -> None:
        """Save the model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, f"{path}_ppo.pt")
    
    def load(self, path: str) -> None:
        """Load the model."""
        checkpoint = torch.load(f"{path}_ppo.pt")
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 