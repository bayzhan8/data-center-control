# Data Center Control System

This project implements a Markov Decision Process (MDP) model for controlling a data center with two types of servers (A and B) and two types of jobs (small and large). The system optimizes server configuration and job assignments to minimize costs while maximizing job completion rewards.

## Features

- Two types of servers: A (for small jobs) and B (for both small and large jobs)
- Two types of jobs: small and large
- Queue management for both job types
- Cost optimization including:
  - Energy costs
  - Switching costs
  - Queue costs
  - Drop costs
  - Completion rewards

## Mathematical Formulation

The system is modeled as a Markov Decision Process with:

- State space: (a_t, b_t, q^S_t, q^L_t)
  - a_t: Number of active A servers
  - b_t: Number of active B servers
  - q^S_t: Number of small jobs in queue
  - q^L_t: Number of large jobs in queue

- Action space: (a^on_{t+1}, b^on_{t+1}, x^S_A, x^S_B, x^L_B)
  - a^on_{t+1}: Number of A servers to activate
  - b^on_{t+1}: Number of B servers to activate
  - x^S_A: Small jobs assigned to A servers
  - x^S_B: Small jobs assigned to B servers
  - x^L_B: Large jobs assigned to B servers

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd data-center-control
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main components are:

- `environment.py`: Implements the MDP environment
- `config.py`: Contains system configuration parameters

Example usage:
```python
from environment import DataCenterEnv
from config import Config

# Create environment
env = DataCenterEnv(Config())

# Example step
state = env.reset()
action = (1, 1, 0, 0, 0)  # Example action
next_state, reward, done, _, info = env.step(action)
```

## Solution Approaches

The system can be solved using:
1. Value Iteration
2. Policy Iteration
3. Reinforcement Learning

## License

[Your chosen license] 