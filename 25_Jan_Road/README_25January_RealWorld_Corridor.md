# Real-World Analysis: 25 January Corridor Network

This document presents the theoretical interpretation and empirical behaviour of the **25 January real-world corridor network** under three traffic-demand scenarios using Fixed-Time control, Tabular Q-Learning, DQN, and multi-agent variants.

Unlike the synthetic two-junction environment, the 25 January corridor contains real-world geometric and operational complexity, including irregular intersection spacing, asymmetric inflows, varying turning movements, and naturally coupled queue spillback between neighbouring intersections. These characteristics create a substantially more difficult control problem and provide a more realistic benchmark for evaluating reinforcement-learning traffic signal controllers.

The experiments compare:

- Fixed-Time baseline control
- Single-Agent Q-Learning
- Single-Agent DQN
- Multi-Agent Q-Learning
- Multi-Agent DQN (MADQN)

All controllers were evaluated under low, medium, and high traffic demand using SUMO-based microscopic traffic simulation.

---

# Low Demand Scenario

## Theoretical Context

Under low demand, the 25 January corridor operates below its physical capacity. Congestion is limited, and the dominant inefficiency comes from unnecessary stopping caused by rigid phase scheduling. In a real-world corridor, this problem becomes more complex than in a synthetic network because traffic arrivals are irregular and influenced by unequal side-street demand, varying block lengths, and naturally formed platoons.

The reinforcement-learning controllers attempt to minimise these unnecessary stops by dynamically allocating green time only to approaches that currently require service. The learning challenge at this demand level is therefore not queue recovery but accurate recognition of sparse and stochastic arrival patterns.

Because the corridor contains multiple interacting intersections, even low demand can create short-lived queue propagation between nearby junctions. Multi-agent controllers benefit from local responsiveness, allowing each intersection to react independently to small fluctuations in demand without waiting for a globally fixed cycle.

## Observed Behaviour

The simulation results show that all RL-based controllers substantially reduced cumulative delay relative to Fixed-Time control under low demand.

Observed trends from the results include:

- Fixed-Time control produced the highest cumulative delay due to unnecessary red phases on lightly loaded approaches.
- Single-Agent and Multi-Agent RL controllers both improved operational efficiency.
- DQN-based controllers showed smoother adaptation to fluctuating arrivals because continuous state representations preserved finer queue information.
- Multi-agent approaches achieved slightly better responsiveness by allowing intersections to react locally.

## TTC Behaviour

Time-To-Collision (TTC) values remained high under low demand across all controllers, indicating safe traffic conditions with minimal close-following interactions.

Approximate observations from the TTC analysis:

- Mean TTC values remained near the safe default region.
- Only a very small percentage of samples fell below critical TTC thresholds.
- RL controllers maintained safety performance while reducing delay.

This indicates that adaptive signal control improved efficiency without increasing collision risk in sparse traffic conditions.

---

# Medium Demand Scenario

## Theoretical Context

At medium demand, the 25 January corridor enters a partially saturated operating regime where queue interactions become persistent rather than occasional. Real-world traffic platoons begin to arrive continuously, and poor phase allocation produces visible queue growth along the corridor.

The control problem becomes significantly harder than in low demand because the agent must balance:

- extending green time long enough to discharge active queues,
- preventing starvation of opposing approaches,
- avoiding queue spillback into upstream intersections.

In the real-world corridor, this challenge is amplified by asymmetric geometry and unequal traffic loading between intersections. A locally optimal decision at one junction may negatively affect downstream traffic progression.

The multi-agent CTDE architecture becomes particularly valuable at this demand level because each intersection can react to local congestion while still learning cooperative discharge patterns during training.

## Observed Behaviour

The medium-demand results showed the clearest advantage of reinforcement-learning control relative to Fixed-Time operation.

Key observations include:

- Fixed-Time control generated steadily increasing queue accumulation across the corridor.
- RL controllers reduced cumulative delay and queue growth substantially.
- Q-Learning performed competitively because discretised congestion states were still sufficient to represent most operating conditions.
- DQN controllers showed improved stability in handling varying queue magnitudes and fluctuating arrivals.
- Multi-agent coordination improved traffic progression between neighbouring intersections.

The real-world geometry amplified the value of adaptive control because fixed cycles could not efficiently respond to unequal directional demand.

## TTC Behaviour

At medium demand, TTC values decreased significantly compared with low demand because vehicle-following interactions became more frequent.

Observed behaviour included:

- Most TTC samples shifted toward lower values due to increased density.
- Differences between controllers became smaller than the differences observed in delay metrics.
- Traffic density rather than signal policy became the dominant factor governing TTC behaviour.

The results indicate that RL controllers improved operational efficiency while maintaining safety characteristics comparable to Fixed-Time control.

---

# High Demand Scenario

## Theoretical Context

Under high demand, the 25 January corridor approaches or exceeds saturation conditions. Queue spillback between intersections becomes persistent, and the traffic network operates in a strongly coupled state where congestion at one junction directly affects neighbouring intersections.

At this level, the traffic signal controller is no longer solving a simple phase-allocation problem. Instead, the controller must continuously triage competing congested approaches while attempting to prevent gridlock propagation through the corridor.

The high-demand regime exposes important differences between controller architectures:

- Fixed-Time control cannot adapt to rapidly changing queue growth.
- Tabular Q-Learning suffers from state-compression limitations because large queues are grouped into coarse bins.
- DQN benefits from continuous state representations that preserve congestion gradients.
- Multi-agent systems scale more effectively because local intersections can respond independently to nearby congestion.

The real-world corridor magnifies these effects because irregular junction spacing and non-uniform inflows create highly dynamic queue interactions.

## Observed Behaviour

The simulation results confirmed that Fixed-Time control performed worst under saturation.

Key trends observed from the experimental data:

- Fixed-Time control produced extremely large cumulative delay and queue growth.
- RL-based controllers significantly reduced congestion escalation.
- Multi-Agent Q-Learning and MADQN consistently outperformed centralised single-agent approaches in congestion management.
- DQN-based methods demonstrated improved handling of severe queue states due to continuous feature representation.
- Multi-agent coordination became increasingly valuable as queue spillback intensified.

The results show that adaptive RL control prevented the uncontrolled delay growth observed under fixed-cycle operation.

## TTC Behaviour

Under high demand, TTC values dropped substantially across all controllers because traffic density became extremely high.

Observed TTC behaviour included:

- Mean TTC values concentrated around low values typical of saturated traffic.
- Nearly all vehicles operated in close-following conditions.
- Differences between controllers became minimal because TTC was dominated primarily by physical density constraints.

This confirms that under saturation, signal-control policy has limited influence on microscopic following distances relative to overall traffic density.

---

# Cross-Demand Analysis

The 25 January real-world corridor results demonstrate several important findings across all traffic conditions.

## 1. Reinforcement Learning Outperformed Fixed-Time Control

Across low, medium, and high demand scenarios, RL-based traffic signal controllers consistently reduced cumulative delay and queue growth relative to the Fixed-Time baseline.

The advantage became increasingly pronounced as demand increased because adaptive controllers could react dynamically to changing traffic conditions while Fixed-Time control remained rigid.

## 2. Multi-Agent Coordination Improved Scalability

Multi-agent architectures showed clear advantages in the real-world corridor because local intersections experienced different congestion patterns simultaneously.

Distributed decision-making improved:

- local responsiveness,
- queue recovery,
- spillback mitigation,
- corridor-wide traffic progression.

These improvements became most significant under high demand where intersection interactions were strongest.

## 3. DQN and Q-Learning Each Showed Different Strengths

The experiments did not establish a universally dominant RL architecture.

Observed behaviour suggests:

- Tabular Q-Learning remained highly competitive at moderate congestion levels.
- DQN handled complex and highly saturated traffic states more effectively.
- MADQN provided the strongest scalability under severe congestion.
- Controller selection depends on the target operating regime, computational cost, and desired optimisation objective.

---

# Overall Conclusion

The 25 January corridor experiments demonstrate that reinforcement-learning traffic signal control can substantially improve traffic efficiency in realistic urban networks.

Compared with Fixed-Time operation, RL controllers:

- reduced cumulative delay,
- reduced queue accumulation,
- improved adaptability to fluctuating traffic demand,
- maintained comparable TTC safety behaviour.

The results also confirm that multi-agent reinforcement learning is especially valuable in real-world urban corridors where congestion interactions between neighbouring intersections dominate network behaviour.

These findings support the feasibility of deploying adaptive RL-based traffic control strategies in complex urban environments where traditional fixed-cycle control cannot respond effectively to dynamic traffic conditions.
