# Theoretical Analysis: Two-Junction Synthetic Network

This document covers the theoretical background and empirical findings for the **Synthetic Two-Junction Corridor** across all three demand scenarios. For real-world results, see the 25 January Corridor analysis.

The two-junction network is a controlled, symmetrical SUMO environment with no real-world geometric noise. It serves as the primary algorithmic testbed — isolating coordination behaviour so that differences between controllers reflect learning architecture rather than infrastructure complexity.

**Network configuration:**
- Turning proportions: 70% straight, 20% left, 10% right
- 24 lane-area detectors across all incoming edges
- 2 traffic lights (TLS), each managed by an independent agent in the multi-agent setting
- Decision interval: every simulation step; SMDP minimum-green lock of 30 steps (3 seconds at 0.10 s step-length)

---

## Low Demand

### Theoretical Context

Under low demand (~42 vehicles/hour/lane), the network operates well below capacity. The dominant inefficiency is not congestion but **unnecessary stopping**: a fixed-time controller serves empty or sparsely occupied approaches on a rigid cycle, forcing arriving vehicles to wait for phases they do not need. The theoretical challenge for RL agents at this demand level is therefore not queue dissipation but **phase-allocation efficiency** — learning to skip or shorten service to unoccupied approaches and extend green to the active stream.

Because vehicle arrivals follow a Poisson process, the RL agents must also learn to tolerate stochastic gaps in demand without over-switching phases. The SMDP lock prevents micro-flickering, but the agent must still distinguish between a genuinely empty approach and a momentary inter-arrival gap. This makes the exploration phase particularly important: early random actions may cause unnecessary switches that increase delay before the agent learns to maintain green for the dominant stream.

Webster's delay model, which assumes deterministic uniform arrivals, overestimates delay under these stochastic low-demand conditions. SUMO simulation consistently produces lower delays than the Webster estimate at low and medium demand, confirming that the simulator captures the irregularity of real vehicle arrivals rather than a worst-case theoretical pattern.

**TTC behaviour:** Under low demand, RL agents maintained high mean TTC values (close to the 50 s safe default) and reduced the proportion of non-default TTC samples relative to Fixed-Time control. This confirms that adaptive controllers improve operational efficiency without introducing additional safety conflicts when traffic is sparse.

---

## Medium Demand

### Theoretical Context

At medium demand (~350 vehicles/hour/lane), the network operates at a meaningful fraction of its capacity. Vehicle platoons arrive with sufficient regularity that **poor phase allocation produces sustained, visible congestion** — not merely occasional unnecessary stops. The fixed-time controller's inability to extend green for loaded approaches or cut short service to lighter approaches causes growing platoon queues and increasing cumulative delay over the 10,000-step horizon.

The theoretical advantage of RL at this demand level is **dynamic green extension**: the agent learns to hold green on an approach while vehicles are still queued and discharge has not yet completed, then switch only once the queue has cleared or the opposing approach has reached a critical threshold. This requires the agent to balance two competing risks — switching too early (leaving vehicles stranded) and switching too late (allowing the opposing approach to saturate).

Tabular Q-Learning can represent this trade-off adequately when the discretised queue bins (0, 1–9, 10–18, 19+) capture the dominant congestion states. DQN processes normalised continuous queue counts and can detect finer gradients in congestion growth, but this added resolution may not translate to better delay performance if the medium-demand state dynamics are sufficiently captured by coarse binning.

The coordination challenge also becomes relevant at this level: in the two-intersection corridor, a green extension at junction 1 directly affects the arrival pattern at junction 2. An agent that is greedy about local queue minimisation may discharge a platoon into a red phase at the downstream intersection, transferring delay rather than eliminating it. The multi-agent CTDE structure allows both agents to learn coordinated discharge patterns, which the single-agent architecture cannot replicate.

**TTC behaviour:** Mean TTC values at medium demand dropped to approximately 2 s across all controllers, with roughly 97–98% of samples falling below the 50 s safe default. At this density, close vehicle-following interactions increase substantially and are governed more by traffic volume than by signal policy. Differences between controllers in TTC metrics were small, indicating that medium-demand safety behaviour is largely capacity-driven.

---

## High Demand

### Theoretical Context

At high demand (840–1,400 vehicles/hour/lane, scaled to push the network toward saturation), the two-junction network enters a regime where the degree of saturation X exceeds 1.0. Webster's delay model breaks down entirely at X > 1 — the formula produces infinite or invalid results — and the SUMO simulation confirms finite but extreme delay values for the fixed-time controller (cumulative delay approaching 68.7 million seconds over 10,000 steps).

The theoretical challenge for RL agents under saturation is fundamentally different from lower demand levels. Queue lengths no longer oscillate between zero and moderate values; instead, **queues grow persistently on every approach**. The agent must learn to triage: prioritise the approach with the largest queue or longest waiting vehicles, discharge it as completely as possible within a green phase, then switch before the opposing approach reaches gridlock. This requires the agent to reason about **queue growth rates and residual capacity**, not just instantaneous queue counts.

Tabular Q-Learning faces a structural limitation under high demand: the discretisation bins (0, 1–9, 10–18, 19+) compress all saturation states above 19 vehicles into a single bin, losing information about whether a queue contains 20 or 200 vehicles. This compression can cause the agent to apply similar actions in qualitatively different saturation states. DQN's continuous normalised inputs preserve this gradient, allowing the agent to distinguish between moderately saturated and severely saturated approaches.

However, the degree to which this advantage materialises depends on whether the agent has experienced enough high-queue states during training to form reliable Q-value estimates across the continuous input space. Under a fully online single-episode training protocol, both agents learn simultaneously on the same trajectory — DQN's advantage depends on whether its neural network has generalised sufficiently by the time the most extreme saturation states are encountered.

The transition from single-agent to multi-agent control is most consequential under high demand. A centralised single agent observing the full joint state must learn a policy over a state-action space that grows exponentially with the number of intersections. Under high demand, the state distributions at the two junctions become coupled through queue spillback — a saturated junction 2 can prevent vehicles from clearing junction 1, and vice versa. The CTDE multi-agent architecture allows each agent to respond to this local pressure independently while learning coordination patterns during training.

**TTC behaviour:** Mean TTC values under high demand fell to approximately **1.5 s** across all controllers, with approximately 98% of samples below the 50 s default and 98% below 3 s. This reflects the extremely high vehicle density and near-continuous close-following interactions under saturation. Differences between controllers were negligible, confirming that TTC at saturation is governed entirely by physical capacity constraints and that signal policy has minimal influence on collision risk when the network is fundamentally overloaded.

---

## Cross-Demand Synthesis

The two-junction results establish three consistent findings across the full demand range:

**1. RL-based control outperforms Fixed-Time control at every demand level.** The magnitude of improvement scales with demand — the largest relative gains appear at medium demand, where rigid fixed cycles are most misaligned with actual traffic patterns, and at high demand, where RL prevents the exponential delay growth that fixed timing cannot contain.

**2. Multi-agent coordination adds value, and most under high demand.** Distributing control decisions across local agents consistently reduced cumulative delay compared with centralised single-agent control, with the improvement most pronounced at saturation — where upstream and downstream queue coupling makes centralised optimisation over the joint state-action space intractable.

**3. Neither Tabular Q-Learning nor DQN/MADQN dominates universally.** Tabular QL was superior in delay minimisation under medium and (single-agent) high demand. DQN held a modest queue advantage in most conditions and showed stronger performance in delay under low demand in the multi-agent setting. The gap between the two architectures effectively closed under multi-agent high-demand saturation, where both converged to similar triage policies. Controller selection should therefore be guided by the target metric, the expected demand range, and the acceptable computational cost.
