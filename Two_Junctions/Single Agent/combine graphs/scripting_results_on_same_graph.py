import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
fixed = pd.read_csv("Baseline_Fixed_result.csv")
q_delay = pd.read_csv("SingleAgent_QLearning_Delay_results.csv")
q_queue = pd.read_csv("SingleAgent_QLearning_Queue_results.csv")
deep_q_delay = pd.read_csv("SingleAgent_DQN_Delay_results.csv")
deep_q_queue = pd.read_csv("SingleAgent_DQN_Queue_results.csv")

# -----------------------------
# 1️. Cumulative Delay Graph
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(fixed["step"], fixed["cum_delay"], label="Fixed Timing")
plt.plot(q_delay["step"], q_delay["cum_delay"], label="Q-Learning (Delay Reward)")
plt.plot(q_delay["step"], deep_q_delay["cum_delay"], label="Deep_Q-Learning (Delay Reward)")

plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Delay Reward")
plt.title("Cumulative Delay Comparison")
plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# 2️. Cumulative Queue Graph
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(fixed["step"], fixed["cum_queue"], label="Fixed Timing")
plt.plot(q_queue["step"], q_queue["cum_queue"], label="Q-Learning (Queue Reward)")
plt.plot(q_queue["step"], deep_q_queue["cum_queue"], label="Deep-Q-Learning (Queue Reward)")

plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Queue Reward")
plt.title("Cumulative Queue Comparison")
plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# 3️. Delay Over Steps
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(fixed["step"], fixed["delay"], label="Fixed Timing")
plt.plot(q_delay["step"], q_delay["delay"], label="Q-Learning (Delay Reward)")
plt.plot(q_delay["step"], deep_q_delay["delay"], label="Deep-Q-Learning (Delay Reward)")

plt.xlabel("Simulation Step")
plt.ylabel("Total Delay")
plt.title("Total Delay Over Simulation Steps")
plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# 4. Queue Over Steps
# -----------------------------
plt.figure(figsize=(10,6))

plt.plot(fixed["step"], fixed["queue"], label="Fixed Timing")
plt.plot(q_queue["step"], q_queue["queue"], label="Q-Learning (Queue Reward)")
plt.plot(q_queue["step"], deep_q_queue["queue"], label="Deep-Q-Learning (Queue Reward)")

plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Queue Length Over Simulation Steps")
plt.legend()
plt.grid(True)

plt.show()
