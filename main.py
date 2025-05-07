import argparse
import json
from typing import Dict, Any
import numpy as np
from config import DataCenterConfig
from environment import DataCenterEnv
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from agents.rl_agent import QLearningAgent
from simulator import Simulator

def run_experiment(agent_type: str, config: DataCenterConfig) -> Dict[str, Any]:
    """Run experiment with specified agent type."""
    # Create environment
    env = DataCenterEnv(config)
    
    # Create agent
    if agent_type == 'vi':
        agent = ValueIterationAgent(env, config)
    elif agent_type == 'pi':
        agent = PolicyIterationAgent(env, config)
    elif agent_type == 'rl':
        agent = QLearningAgent(env, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Train agent
    print(f"Training {agent_type} agent...")
    training_metrics = agent.train()
    print(f"Training completed. Metrics: {training_metrics}")
    
    # Create simulator
    simulator = Simulator(env, agent, config)
    
    # Run evaluation episodes
    print(f"Running evaluation episodes for {agent_type} agent...")
    eval_metrics = simulator.run_multiple_episodes(config.eval_episodes)
    print(f"Evaluation completed. Metrics: {eval_metrics}")
    
    return {
        'agent_type': agent_type,
        'training_metrics': training_metrics,
        'eval_metrics': eval_metrics
    }

def main():
    parser = argparse.ArgumentParser(description='Data Center Control Experiment')
    parser.add_argument('--agent', type=str, choices=['vi', 'pi', 'rl', 'all'],
                      default='all', help='Agent type to run')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DataCenterConfig(**config_dict)
    else:
        config = DataCenterConfig()
    
    # Run experiments
    if args.agent == 'all':
        agent_types = ['vi', 'pi', 'rl']
    else:
        agent_types = [args.agent]
    
    results = {}
    for agent_type in agent_types:
        print(f"\nRunning experiment for {agent_type} agent...")
        results[agent_type] = run_experiment(agent_type, config)
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print("\nComparison of agents:")
    print("-" * 120)
    print(f"{'Agent':<10} {'Avg Reward':<15} {'Std Reward':<15} {'Energy Cost':<15} {'Queue Cost':<15} {'Drop Cost':<15} {'Dropped Jobs':<15}")
    print("-" * 120)
    for agent_type in agent_types:
        metrics = results[agent_type]['eval_metrics']
        print(f"{agent_type:<10} "
              f"{metrics['avg_reward']:<15.2f} "
              f"{metrics['std_reward']:<15.2f} "
              f"{metrics['avg_energy_cost']:<15.2f} "
              f"{metrics['avg_queue_cost']:<15.2f} "
              f"{metrics['avg_drop_cost']:<15.2f} "
              f"{metrics['avg_dropped_jobs']:<15.2f}")

if __name__ == '__main__':
    main() 