import argparse
from typing import Dict, Any, Tuple
from environment import DataCenterEnv
from config import DataCenterConfig
from agents.base_agent import BaseAgent
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from agents.rl_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np

def parse_args() -> str:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data Center Control System')
    parser.add_argument('--agent', type=str, choices=['vi', 'pi', 'rl', 'dqn', 'ppo', 'all'],
                      default='all', help='Agent type to run (vi=Value Iteration, pi=Policy Iteration, rl=Q-learning, dqn=Deep Q-Network, ppo=Proximal Policy Optimization)')
    return parser.parse_args().agent

def train_agent(agent_type: str, env: DataCenterEnv, config: DataCenterConfig) -> Tuple[BaseAgent, Dict[str, Any]]:
    """Train an agent and return both the agent and training results."""
    if agent_type == 'vi':
        agent = ValueIterationAgent(env, config)
        agent.train()
        return agent, {'episode_rewards': []}  # VI doesn't have episode rewards
    elif agent_type == 'pi':
        agent = PolicyIterationAgent(env, config)
        agent.train()
        return agent, {'episode_rewards': []}  # PI doesn't have episode rewards
    elif agent_type == 'rl':
        agent = QLearningAgent(env, config)
        results = agent.train()
        return agent, results
    elif agent_type == 'dqn':
        agent = DQNAgent(env, config)
        results = agent.train()
        return agent, results
    elif agent_type == 'ppo':
        agent = PPOAgent(env, config)
        results = agent.train()
        return agent, results
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def evaluate_agent(agent_type: str, agent: Any, env: DataCenterEnv, config: DataCenterConfig) -> Dict[str, float]:
    """Evaluate a trained agent and return the metrics."""
    simulator = Simulator(env, agent, config)
    metrics = simulator.run_multiple_episodes(config.eval_episodes)
    
    print(f"\n{agent_type.upper()} Results:")
    print(f"Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Average Energy Cost: {metrics['avg_energy_cost']:.2f} ± {metrics['std_energy_cost']:.2f}")
    print(f"Average Queue Cost: {metrics['avg_queue_cost']:.2f} ± {metrics['std_queue_cost']:.2f}")
    print(f"Average Dropped Jobs: {metrics['avg_dropped_jobs']:.2f} ± {metrics['std_dropped_jobs']:.2f}")
    
    return metrics

def plot_learning_curve(results: Dict[str, Any], agent_type: str):
    """Plot learning curve for RL agents."""
    episode_rewards = results['episode_rewards']
    
    # Calculate moving average
    window_size = 10
    moving_avg = np.convolve(episode_rewards, 
                            np.ones(window_size)/window_size, 
                            mode='valid')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Raw Rewards')
    plt.plot(moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.title(f'{agent_type} Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis to show all episodes
    plt.xlim(0, len(episode_rewards))
    
    # Save plot
    plt.savefig(f'{agent_type.lower()}_learning_curve.png')
    plt.close()

def main():
    # Parse arguments and setup
    agent_type = parse_args()
    config = DataCenterConfig()
    env = DataCenterEnv(config)
    
    # Train and evaluate agents
    if agent_type == 'all':
        agent_types = ['vi', 'pi', 'rl', 'dqn', 'ppo']
    else:
        agent_types = [agent_type]
    
    for agent_type in agent_types:
        # Train agent
        agent, results = train_agent(agent_type, env, config)
        
        # Evaluate agent
        evaluate_agent(agent_type, agent, env, config)
        
        # Plot learning curve for RL agents
        if agent_type in ['rl', 'dqn', 'ppo']:
            plot_learning_curve(results, agent_type.upper())
        
        # Save trained agent
        agent.save(f'{agent_type}_model.pkl')
        print(f"Model saved to {agent_type}_model.pkl")

if __name__ == "__main__":
    main() 