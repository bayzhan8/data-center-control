This project implements various reinforcement learning algorithms to optimize data center resource allocation and job scheduling. The system manages two types of servers (Type A and Type B) and handles both small and large jobs with different queue capacities.

## Problem Description

The data center control system must:
- Manage server allocation (turning servers on/off)
- Schedule jobs to appropriate servers
- Handle job queues for both small and large jobs
- Minimize energy costs while maximizing job completion

### State Space
- Number of active Type A servers (0 to N_A)
- Number of active Type B servers (0 to N_B)
- Number of small jobs in queue (0 to max_queue_small)
- Number of large jobs in queue (0 to max_queue_large)

### Actions
- Allocate servers (turn on/off)
- Assign jobs to servers
- Handle job queues

### Rewards/Costs
- Energy costs for active servers
- Switching costs for changing server states
- Queue costs for waiting jobs
- Penalties for dropped jobs
- Rewards for completed jobs

## Implemented Algorithms

1. **Value Iteration (VI)**
   - Dynamic programming approach
   - Finds optimal policy through iterative value function updates
   - Guaranteed convergence to optimal solution

2. **Policy Iteration (PI)**
   - Alternates between policy evaluation and improvement
   - More efficient than value iteration in practice
   - Also guaranteed to converge to optimal solution

3. **Q-Learning (RL)**
   - Model-free reinforcement learning
   - Parameters:
     - Episodes: 2000
     - Learning rate: 0.01
     - Discount factor: 0.99
     - Epsilon: 0.5 (exploration rate)
     - Min episodes: 1500
     - Max no improvement: 300

4. **Deep Q-Network (DQN)**
   - Neural network-based Q-learning
   - Parameters:
     - Episodes: 2000
     - Learning rate: 0.005
     - Batch size: 128
     - Memory size: 10000
     - Target update: every 5 episodes
     - Epsilon decay: 0.995

5. **Proximal Policy Optimization (PPO)**
   - Policy gradient method with clipping
   - Parameters:
     - Episodes: 2000
     - Learning rate: 0.0005
     - Clip ratio: 0.2
     - Value coefficient: 0.5
     - Entropy coefficient: 0.05
     - Epochs per update: 8

## Configuration

The system is configured through `config.py` with the following key parameters:

```python
# Server configuration
N_A = 5  # Type A servers
N_B = 6  # Type B servers

# Queue configuration
max_queue_small = 5  # Small job queue
max_queue_large = 7  # Large job queue

# Cost parameters
c_A = 1.0  # Energy cost per A server
c_B = 2.0  # Energy cost per B server
c_switch = 0.5  # Server state change cost
c_queue = 0.1  # Queue cost per job
c_drop = 5.0  # Job drop penalty

# Job completion rewards
c_complete_small = 2.0  # Small job reward
c_complete_large = 4.0  # Large job reward

# Job arrival probabilities
p_S = 0.3  # Small job probability
p_L = 0.2  # Large job probability
```

## Usage

Run the system with different algorithms:

```bash
# Run specific algorithm
python main.py --agent vi    # Value Iteration
python main.py --agent pi    # Policy Iteration
python main.py --agent rl    # Q-learning
python main.py --agent dqn   # Deep Q-Network
python main.py --agent ppo   # PPO

# Run all algorithms
python main.py --agent all
```

## Output

For each algorithm, the system provides:
1. Training progress with average rewards
2. Final evaluation metrics:
   - Average reward
   - Energy costs
   - Queue costs
   - Dropped jobs
3. Learning curve plot (for RL algorithms)
4. Saved model file

## Requirements

- Python 3.8+
- NumPy
- PyTorch
- Matplotlib
- tqdm

## Project Structure

```
.
├── agents/
│   ├── base_agent.py
│   ├── value_iteration.py
│   ├── policy_iteration.py
│   ├── rl_agent.py
│   ├── dqn_agent.py
│   └── ppo_agent.py
├── environment.py
├── config.py
├── main.py
├── simulator.py
└── README.md
```

## Future Improvements

1. Implement more advanced RL algorithms
2. Add parallel training support
3. Optimize hyperparameters further
4. Add more sophisticated reward shaping
5. Implement multi-agent approaches 
