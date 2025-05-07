from typing import Dict, List, Any, Tuple
import numpy as np
from environment import DataCenterEnv
from agents.base_agent import BaseAgent
from config import DataCenterConfig

class Simulator:
    def __init__(self, env: DataCenterEnv, agent: BaseAgent, config: DataCenterConfig):
        self.env = env
        self.agent = agent
        self.config = config
    
    def run_episode(self, max_steps: int = 10000) -> Dict[str, Any]:
        """Run a single episode and collect metrics."""
        state, _ = self.env.reset()
        total_reward = 0
        total_energy_cost = 0
        total_switch_cost = 0
        total_queue_cost = 0
        total_drop_cost = 0
        total_dropped_jobs = 0
        queue_lengths = []
        active_servers = []
        
        for step in range(max_steps):
            # Get action from agent
            action = self.agent.act(state)
            a_next, b_next, x_s_a, x_s_b, x_l_b = action
            a_t, b_t, q_s, q_l = state
            
            # Take step in environment
            next_state, reward, done, _, info = self.env.step(action)
            
            # Calculate costs
            energy_cost = (self.config.c_A * a_next + 
                         self.config.c_B * b_next)
            switch_cost = (self.config.c_switch * 
                         (abs(a_next - a_t) + abs(b_next - b_t)))
            queue_cost = self.config.c_queue * (q_s + q_l)
            drop_cost = self.config.c_drop * info['dropped_jobs']
            
            # Update metrics
            total_reward += reward
            total_energy_cost += energy_cost
            total_switch_cost += switch_cost
            total_queue_cost += queue_cost
            total_drop_cost += drop_cost
            total_dropped_jobs += info['dropped_jobs']
            queue_lengths.append(q_s + q_l)
            active_servers.append(a_next + b_next)
            
            state = next_state
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'total_energy_cost': total_energy_cost,
            'total_switch_cost': total_switch_cost,
            'total_queue_cost': total_queue_cost,
            'total_drop_cost': total_drop_cost,
            'total_dropped_jobs': total_dropped_jobs,
            'avg_queue_length': np.mean(queue_lengths),
            'max_queue_length': max(queue_lengths),
            'avg_active_servers': np.mean(active_servers),
            'steps': len(queue_lengths)
        }
    
    def run_multiple_episodes(self, num_episodes: int = 1000) -> Dict[str, List[Any]]:
        """Run multiple episodes and collect aggregate metrics."""
        episode_metrics = []
        
        # Run more episodes for better statistics
        for _ in range(num_episodes):
            metrics = self.run_episode()
            episode_metrics.append(metrics)
        
        # Calculate statistics
        rewards = [m['total_reward'] for m in episode_metrics]
        energy_costs = [m['total_energy_cost'] for m in episode_metrics]
        switch_costs = [m['total_switch_cost'] for m in episode_metrics]
        queue_costs = [m['total_queue_cost'] for m in episode_metrics]
        drop_costs = [m['total_drop_cost'] for m in episode_metrics]
        dropped_jobs = [m['total_dropped_jobs'] for m in episode_metrics]
        queue_lengths = [m['avg_queue_length'] for m in episode_metrics]
        active_servers = [m['avg_active_servers'] for m in episode_metrics]
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_energy_cost': np.mean(energy_costs),
            'std_energy_cost': np.std(energy_costs),
            'avg_switch_cost': np.mean(switch_costs),
            'std_switch_cost': np.std(switch_costs),
            'avg_queue_cost': np.mean(queue_costs),
            'std_queue_cost': np.std(queue_costs),
            'avg_drop_cost': np.mean(drop_costs),
            'std_drop_cost': np.std(drop_costs),
            'avg_dropped_jobs': np.mean(dropped_jobs),
            'std_dropped_jobs': np.std(dropped_jobs),
            'avg_queue_length': np.mean(queue_lengths),
            'std_queue_length': np.std(queue_lengths),
            'avg_active_servers': np.mean(active_servers),
            'std_active_servers': np.std(active_servers),
            'max_queue_length': max([m['max_queue_length'] for m in episode_metrics]),
            'avg_steps': np.mean([m['steps'] for m in episode_metrics])
        } 