import numpy as np
import matplotlib.pyplot as plt
from config import DataCenterConfig
from environment import DataCenterEnv
from agents.value_iteration import ValueIterationAgent
from agents.policy_iteration import PolicyIterationAgent
from agents.rl_agent import QLearningAgent

def analyze_policy(agent, env, state_space):
    """Analyze policy for different queue states."""
    policy_analysis = {}
    
    # For each queue state
    for q_s in range(env.config.max_queue_small + 1):
        for q_l in range(env.config.max_queue_large + 1):
            state = (0, 0, q_s, q_l)  # Start with no active servers
            action = agent.act(state)
            a_next, b_next, x_s_a, x_s_b, x_l_b = action
            
            policy_analysis[(q_s, q_l)] = {
                'a_next': a_next,
                'b_next': b_next,
                'x_s_a': x_s_a,
                'x_s_b': x_s_b,
                'x_l_b': x_l_b,
                'total_servers': a_next + b_next
            }
    
    return policy_analysis

def print_policy_summary(policy, agent_name):
    """Print summary statistics for a policy."""
    total_servers = [action['total_servers'] for action in policy.values()]
    a_servers = [action['a_next'] for action in policy.values()]
    b_servers = [action['b_next'] for action in policy.values()]
    
    print(f"\n{agent_name} Policy Summary:")
    print(f"Average total servers: {np.mean(total_servers):.2f}")
    print(f"Average A servers: {np.mean(a_servers):.2f}")
    print(f"Average B servers: {np.mean(b_servers):.2f}")
    print(f"Max total servers: {max(total_servers)}")
    print(f"Min total servers: {min(total_servers)}")
    
    # Count states where servers are activated
    active_states = sum(1 for action in policy.values() if action['total_servers'] > 0)
    print(f"States with active servers: {active_states}/{len(policy)}")

def visualize_policies(vi_policy, pi_policy, rl_policy, config):
    """Visualize policies for different queue states."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Policy Comparison: Server Activation Decisions')
    
    # Plot total servers for VI
    ax = axes[0, 0]
    data = np.zeros((config.max_queue_small + 1, config.max_queue_large + 1))
    for (q_s, q_l), action in vi_policy.items():
        data[q_s, q_l] = action['total_servers']
    im = ax.imshow(data, cmap='YlOrRd')
    ax.set_title('VI: Total Servers')
    ax.set_xlabel('Large Queue Length')
    ax.set_ylabel('Small Queue Length')
    plt.colorbar(im, ax=ax)
    
    # Plot total servers for PI
    ax = axes[0, 1]
    data = np.zeros((config.max_queue_small + 1, config.max_queue_large + 1))
    for (q_s, q_l), action in pi_policy.items():
        data[q_s, q_l] = action['total_servers']
    im = ax.imshow(data, cmap='YlOrRd')
    ax.set_title('PI: Total Servers')
    ax.set_xlabel('Large Queue Length')
    ax.set_ylabel('Small Queue Length')
    plt.colorbar(im, ax=ax)
    
    # Plot total servers for RL
    ax = axes[1, 0]
    data = np.zeros((config.max_queue_small + 1, config.max_queue_large + 1))
    for (q_s, q_l), action in rl_policy.items():
        data[q_s, q_l] = action['total_servers']
    im = ax.imshow(data, cmap='YlOrRd')
    ax.set_title('RL: Total Servers')
    ax.set_xlabel('Large Queue Length')
    ax.set_ylabel('Small Queue Length')
    plt.colorbar(im, ax=ax)
    
    # Plot difference between VI and RL
    ax = axes[1, 1]
    data = np.zeros((config.max_queue_small + 1, config.max_queue_large + 1))
    for (q_s, q_l) in vi_policy.keys():
        vi_servers = vi_policy[(q_s, q_l)]['total_servers']
        rl_servers = rl_policy[(q_s, q_l)]['total_servers']
        data[q_s, q_l] = rl_servers - vi_servers
    im = ax.imshow(data, cmap='RdBu')
    ax.set_title('Difference: RL - VI')
    ax.set_xlabel('Large Queue Length')
    ax.set_ylabel('Small Queue Length')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png')
    plt.close()

def main():
    # Create environment and config
    config = DataCenterConfig()
    env = DataCenterEnv(config)
    
    # Create and train agents
    print("Training Value Iteration agent...")
    vi_agent = ValueIterationAgent(env, config)
    vi_agent.train()
    
    print("Training Policy Iteration agent...")
    pi_agent = PolicyIterationAgent(env, config)
    pi_agent.train()
    
    print("Training RL agent...")
    rl_agent = QLearningAgent(env, config)
    rl_agent.train()
    
    # Analyze policies
    print("Analyzing policies...")
    vi_policy = analyze_policy(vi_agent, env, config)
    pi_policy = analyze_policy(pi_agent, env, config)
    rl_policy = analyze_policy(rl_agent, env, config)
    
    # Print policy summaries
    print_policy_summary(vi_policy, "Value Iteration")
    print_policy_summary(pi_policy, "Policy Iteration")
    print_policy_summary(rl_policy, "Reinforcement Learning")
    
    # Compare policies
    print("\nKey Policy Differences:")
    print("-" * 80)
    for q_s in range(config.max_queue_small + 1):
        for q_l in range(config.max_queue_large + 1):
            vi_action = vi_policy[(q_s, q_l)]
            pi_action = pi_policy[(q_s, q_l)]
            rl_action = rl_policy[(q_s, q_l)]
            
            # Only print states where policies differ
            if (vi_action['total_servers'] != rl_action['total_servers'] or
                pi_action['total_servers'] != rl_action['total_servers']):
                print(f"\nQueue State ({q_s}, {q_l}):")
                print(f"  VI: {vi_action['total_servers']} servers (A={vi_action['a_next']}, B={vi_action['b_next']})")
                print(f"  PI: {pi_action['total_servers']} servers (A={pi_action['a_next']}, B={pi_action['b_next']})")
                print(f"  RL: {rl_action['total_servers']} servers (A={rl_action['a_next']}, B={rl_action['b_next']})")
    
    # Visualize policies
    print("\nGenerating policy visualization...")
    visualize_policies(vi_policy, pi_policy, rl_policy, config)
    print("Visualization saved as 'policy_comparison.png'")

if __name__ == '__main__':
    main()