# Adaptive Traffic Signal Control Using Multi-Agent Reinforcement Learning

> **Paper:** *Adaptive Traffic Signal Control Using Multi-Agent Reinforcement Learning: A Comparison of Control Strategies*
> **Authors:** Mahmoud Owais et al. — Civil & Electrical Engineering Departments, Assiut University, Egypt
> **Simulator:** SUMO + Python + TraCI

---

## Overview

This project implements and compares three traffic signal control strategies across two road networks and three demand levels using microscopic traffic simulation:

| Strategy | Type |
|---|---|
| Fixed-Time Control | Baseline (no learning) |
| Tabular Q-Learning | Value-based RL |
| Deep Q-Network (DQN / MADQN) | Deep value-based RL |

Each strategy is evaluated in both **Single-Agent** and **Multi-Agent** configurations, optimising for either **queue minimisation** or **delay minimisation** as the reward signal.

---

## Project Structure

```
project/
│
├── networks/
│   ├── two_junction/               # Synthetic two-intersection corridor
│   │   ├── two_junction.net.xml
│   │   ├── two_junction.sumocfg
│   │   ├── two_junction_high.rou.xml
│   │   ├── two_junction_medium.rou.xml
│   │   └── two_junction_low.rou.xml
│   │
│   └── 25_jan/                     # Real-world digital twin, Assiut, Egypt
│       ├── 25_Jan.net.xml
│       ├── 25_Jan.sumocfg
│       ├── 25_Jan_high.rou.xml
│       ├── 25_Jan_medium.rou.xml
│       └── 25_Jan_low.rou.xml
│
├── controllers/
│   │
│   ├── fixed_time/
│   │   └── base_fixed_time.py               # Fixed-cycle baseline (both networks)
│   │
│   ├── single_agent/
│   │   ├── SingleAgent_DQN_Delay.py         # DQN — delay reward
│   │   ├── SingleAgent_DQN_Queue.py         # DQN — queue reward
│   │   ├── SingleAgent_QLearning_Delay.py   # Tabular Q-Learning — delay reward
│   │   └── SingleAgent_QLearning_Queue.py   # Tabular Q-Learning — queue reward
│   │
│   └── multi_agent/
│       ├── MultiAgent_DQN_Delay.py          # MADQN — delay reward
│       ├── MultiAgent_DQN_Queue.py          # MADQN — queue reward
│       ├── MultiAgent_QLearning_Delay.py    # Multi-Agent Tabular QL — delay reward
│       └── MultiAgent_QLearning_Queue.py    # Multi-Agent Tabular QL — queue reward
│
├── combine graphs/                  # Output CSVs and combined plots
│   ├── *_demand_*_results.csv
│   └── *_demand_*_TTC_results.csv
│
├── THEORY.md                        # Theoretical background and results analysis
└── README.md                        # This file
```

---

## Networks

### Synthetic Two-Junction Corridor
A controlled, symmetrical two-intersection corridor built entirely in SUMO. Used as the primary algorithmic testbed — isolates coordination behaviour without real-world geometric noise.

- Turning proportions: **70% straight / 20% left / 10% right**
- Purpose: clean baseline for comparing learning architectures

### 25 January Corridor (Assiut, Egypt) — Digital Twin
A real-world arterial corridor modelled as a SUMO digital twin from satellite imagery and field data.

- Turning proportions: **82% straight / 12% major turn / 6% minor turn**
- Asymmetric geometry and lane capacities challenge the RL agents with realistic directional imbalance

---

## Demand Scenarios

| Level | Volume | Represents |
|---|---|---|
| Low | ~42 veh/hr/lane | Off-peak (early morning / late night) |
| Medium | ~350 veh/hr/lane | Standard urban arterial operation |
| High | 840–1,400 veh/hr/lane (network-dependent) | Peak congestion / saturation stress test |

Vehicle arrivals follow a **Poisson process**. All scenarios use fixed random seeds for reproducibility.

---

## Controllers

### Fixed-Time Baseline (`base_fixed_time.py`)
Cycles through predefined phase durations regardless of real-time traffic. Used as the performance benchmark that all RL methods must outperform.

### Tabular Q-Learning (Single & Multi-Agent)
Updates a discrete Q-table via the Bellman equation. State is discretised into queue-density bins. Computationally lightweight and highly interpretable.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.9 to 0.99 |
| Exploration start (ε) | 1.0 |
| Exploration minimum | 0.1 |
| Decay rate | 0.992 to 0.9995 |

### DQN / MADQN (Single & Multi-Agent)
A fully connected feedforward neural network approximates the Q-function over normalised, continuous traffic states.

**Architecture:** Input → Dense(256, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Linear output

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.0002 to 0.0005 |
| Discount factor (γ) | 0.9 to 0.99 |
| Replay buffer size | 50,000 to 200,000 |
| Mini-batch size | 64 to 128 |
| Target network soft-update (τ) | 0.002 to 0.01 |
| Update frequency | Every 200 to 500 steps |

### Multi-Agent Coordination: CTDE
All multi-agent controllers use **Centralised Training with Decentralised Execution (CTDE)**. Agents train using shared global information but act independently on local observations — balancing coordination benefits with scalability.

---

## State & Action Space

### State Vector
```
S_raw = [q_1, q_2, ..., q_24, p_1, p_2]
```
Where `q_i` = halting vehicle count on detector `i`, and `p_j` = active phase index of traffic light `j`.

- **Tabular:** queue counts discretised into 4 density bins (0, 1–9, 10–18, 19+)
- **DQN:** queue counts normalised by dividing by 10.0 and clipped to [0, 1]

### Action Space
Binary per intersection at each decision step:
- `0` — Maintain current green phase
- `1` — Switch to next phase in sequence

An **SMDP minimum-green lock** (4–12 seconds) prevents unrealistic phase flickering and aligns with physical signal-transition constraints.

---

## Reward Functions

Two reward variants are implemented per agent type:

**Queue Minimisation:**
```
R_t = - Σ q_i(t)     (sum of halting vehicles across all detectors)
```

**Delay Minimisation:**
```
R_t = - Σ w_j(t)     (sum of accumulated waiting time across all incoming edges)
```

Both formulate the objective as a negative penalty, directing the agent toward minimising congestion.

---

## Performance Metrics

| Metric | Description |
|---|---|
| Instantaneous Delay | Per-step total vehicle waiting time (seconds) |
| Cumulative Delay | Running sum of delay over the full 10,000-step horizon |
| Instantaneous Queue | Per-step total halting vehicle count |
| Cumulative Queue | Running sum of queue accumulation |
| Time-To-Collision (TTC) | Surrogate safety indicator: `TTC = d / (v_follower − v_leader)` when closing |

TTC is recorded every 100 steps. A default safe value of 50 s is assigned when no critical following interaction is detected. Values below **3 s** are treated as unsafe conflicts; below **1.5 s** as critical conflicts.

---

## Running the Code

### Prerequisites

```bash
pip install numpy matplotlib pandas tensorflow traci
```

SUMO must be installed and the `SUMO_HOME` environment variable must be set:

```bash
export SUMO_HOME=/path/to/sumo
```

### Selecting Demand Level

At the top of each controller script, set the demand level:

```python
high   = 0
medium = 1
low    = 2
selected_demand = low    # <-- change here
```

### Running a Controller

```bash
# Example: Multi-Agent DQN with delay reward on 25 January network
python controllers/multi_agent/MultiAgent_DQN_Delay.py
```

To visualise in the SUMO GUI, change the first element of `Sumo_config` from `'sumo'` to `'sumo-gui'`.

### Output Files

Results are saved automatically to `combine graphs/` as CSV files:

```
{demand_level}_{agent_type}_{algorithm}_{reward_type}_results.csv
{demand_level}_{agent_type}_{algorithm}_{reward_type}_TTC_results.csv
```

Columns: `step`, `queue`, `delay`, `cum_delay` (and `min_ttc` for TTC files).

---

**Overall finding:** No single architecture universally dominates. MADQN generalises better in low-demand and queue-dissipation scenarios; Tabular Q-Learning is more competitive under medium- and high-demand conditions, especially in the real-world corridor.

---

## Safety Note

TTC values decrease across all controllers as demand increases — close-following interactions under saturation are governed by physical capacity constraints rather than signal policy. TTC should always be interpreted alongside delay and queue metrics, not in isolation.

---

## Citation

```
Owais, M. et al. (2025). Adaptive Traffic Signal Control Using Multi-Agent Reinforcement
Learning: A Comparison of Control Strategies. Sustainability, 17.
```

---

## License

Submitted under Creative Commons Attribution (CC BY 4.0).
