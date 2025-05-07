import argparse
from typing import Dict, Any, Tuple
from environment import DataCenterEnv
from config import DataCenterConfig
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from agents.rl_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from simulator import Simulator

def parse_args() -> str:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data Center Control System')
    parser.add_argument('--agent', type=str, choices=['vi', 'pi', 'rl', 'dqn', 'all'],
                      default='all', help='Agent type to run (vi=Value Iteration, pi=Policy Iteration, rl=Q-learning, dqn=Deep Q-Network)')
    return parser.parse_args().agent

def train_agent(agent_type: str, env: DataCenterEnv, config: DataCenterConfig) -> Tuple[Any, Dict[str, Any]]:
    """Train a specific agent type and return the agent and training results."""
    if agent_type == 'vi':
        print("Training Value Iteration agent...")
        agent = ValueIterationAgent(env, config)
        results = agent.train()
        print(f"Value Iteration completed in {results['iterations']} iterations")
    
    elif agent_type == 'pi':
        print("\nTraining Policy Iteration agent...")
        agent = PolicyIterationAgent(env, config)
        results = agent.train()
        print(f"Policy Iteration completed in {results['iterations']} iterations")
    
    elif agent_type == 'rl':
        print("\nTraining Q-learning agent...")
        agent = QLearningAgent(env, config)
        results = agent.train()
        print(f"Q-learning completed in {results['episodes_trained']} episodes")
    
    elif agent_type == 'dqn':
        print("\nTraining DQN agent...")
        agent = DQNAgent(env, config)
        results = agent.train()
        print(f"DQN completed in {results['episodes_trained']} episodes")
        print(f"Final epsilon: {results['final_epsilon']:.3f}")
        print(f"Final average reward: {results['final_avg_reward']:.2f}")
    
    return agent, results

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

def main():
    # Parse arguments and setup
    agent_type = parse_args()
    config = DataCenterConfig()
    env = DataCenterEnv(config)
    
    # Train and evaluate agents
    if agent_type == 'all':
        agent_types = ['vi', 'pi', 'rl', 'dqn']
    else:
        agent_types = [agent_type]
    
    for agent_type in agent_types:
        # Train agent
        agent, _ = train_agent(agent_type, env, config)
        
        # Evaluate agent
        evaluate_agent(agent_type, agent, env, config)

if __name__ == "__main__":
    main() 