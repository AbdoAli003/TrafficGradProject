# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

import pandas as pd # saving

# Force Python to run inside the exact folder where this script lives ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
high = 0 
medium = 1
low = 2
selected_demand = low
rou_files = ["RL_high.rou.xml","RL_medium.rou.xml","RL_low.rou.xml"]
selected_route = rou_files[selected_demand]
Sumo_config = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--route-files', selected_route,
    '--step-length', '0.1',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
current_phase = []
stopping_car = None
# Get ALL lane area detector IDs automatically
all_detectors = list(traci.lanearea.getIDList())
# Get ALL incoming edgs with vehicles for all traffic ligts (TLS)
incoming_edges = list(set(
    traci.lane.getEdgeID(lane)
    for tls in traci.trafficlight.getIDList()
    for lane in traci.trafficlight.getControlledLanes(tls)
    if not traci.lane.getEdgeID(lane).startswith(":")
))
tls_ids = list(traci.trafficlight.getIDList())
print("TLS IDs: - SingleAgent_QLearning_Queue.py:61", tls_ids) #to depug
print(incoming_edges) #to depug
print(all_detectors) #to depug

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.99            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 1          # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.992

NUM_TLS = len(tls_ids)

# Generate joint actions for all TLS
ACTIONS = [0,1]

# Q-table dictionary: key = state tuple, value = numpy array of Q-values for each action
Q_table = {}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 120
last_switch_step = -MIN_GREEN_STEPS

# Map each detector to its edge
detector_edge_map = {}

for det in all_detectors:
    lane_id = traci.lanearea.getLaneID(det)
    edge_id = traci.lane.getEdgeID(lane_id)

    if edge_id.startswith(":"):
        continue

    detector_edge_map[det] = edge_id

# unique incoming edges
edge_list = sorted(list(set(detector_edge_map.values())))

# map edges to detectors
edge_detectors = {edge: [] for edge in edge_list}

for det, edge in detector_edge_map.items():
    edge_detectors[edge].append(det)

print("Edge groups: - SingleAgent_QLearning_Queue.py:110", edge_detectors)
# -------------------------
# Step 7: Define Functions
# -------------------------

def inject_breakdown(lane_id,edge_id, duration=100):
    global stopping_car
    """
    Injects a temporary breakdown of a vehicle in the middle of a specified edge.

    edge_id : ID of the edge where breakdown will happen
    duration: Number of simulation steps the vehicle remains stopped
              (100 steps = 10 seconds when step-length = 0.10)
    """
    # Get all vehicles currently on this edge
    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)

    if len(vehicles_on_lane) == 0:
        return  # No vehicle available to break down

    # Select one random vehicle from this edge
    veh_id = random.choice(vehicles_on_lane)

    try:
      if stopping_car == None :
        current_position = traci.vehicle.getLanePosition(veh_id)
        lane_index = traci.vehicle.getLaneIndex(veh_id)
        # Force vehicle to stop for 'duration' steps
        traci.vehicle.setStop(
            vehID=veh_id,
            edgeID=edge_id,
            pos=current_position,
            laneIndex=lane_index,
            duration=duration
        )

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps. - SingleAgent_QLearning_Queue.py:146")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found - SingleAgent_QLearning_Queue.py:150")
        pass  # Ignore errors if vehicle disappears

def get_max_Q_value_of_state(s): #1. Objective Function 1
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):  #2. Constraint 2 
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-len(tls_ids)]) # Exclude the current_phase element
    reward = -float(total_queue) 
    return reward

def get_state():
    global current_phase

    # Get queue from ALL lane area detectors dynamically
    queues = []
    for detector_id in all_detectors:
        q = get_queue_length(detector_id)
        queues.append(q)

    # Get phase for ALL traffic lights
    phases = []
    for tls in tls_ids:
        phase = get_current_phase(tls)
        phases.append(phase)
    current_phase = phases
    # state = queues + phases
    return tuple(queues + phases)


def apply_action(action): #5. Constraint 5
    global last_switch_step, current_simulation_step

    if current_simulation_step - last_switch_step < MIN_GREEN_STEPS:
        return

    is_switched = 0
    # Decode action to binary vector
    for i, tls in enumerate(tls_ids):

        if action == 1:   # if bit is 1 → switch
            is_switched = 1
            program = traci.trafficlight.getAllProgramLogics(tls)[0]
            num_phases = len(program.phases)

            next_phase = (get_current_phase(tls) + 1) % num_phases

            traci.trafficlight.setPhase(tls, next_phase)
    if is_switched == 1:
       last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state): #6. Constraint 6
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    
    
    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_table[old_state][action]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(new_state)
    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state): #7. Constraint 7
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

def compress_state(state):

    num_tls = len(tls_ids)
    queues = state[:-num_tls]
    phases = state[-num_tls:]

    # rebuild detector dictionary
    detector_queue = dict(zip(all_detectors, queues))

    edge_bins = []

    for edge in edge_list:

        total_edge_queue = sum(detector_queue[d] for d in edge_detectors[edge])

        if total_edge_queue == 0:
            q_bin = 0
        elif total_edge_queue <= 9:
            q_bin = 1
        elif total_edge_queue <= 18:
            q_bin = 2
        else:
            q_bin = 3

        edge_bins.append(q_bin)

    return tuple(edge_bins + list(phases))

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepHaltingNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []
delay_history = []


print("\n=== Starting Fully Online Continuous Learning === - SingleAgent_QLearning_Queue.py:272")
episodes = 1
for episode in range(episodes):
  if episode !=0:
        traci.start(Sumo_config)
        traci.gui.setSchema("View #0", "real world")
  step_history = []
  reward_history = []
  queue_history = []
  delay_history = []
  cumulative_queue_reward = 0.0
  EPSILON = 1.0 - episode/episodes
  print(f"start of episode {episode} - SingleAgent_QLearning_Queue.py:284")
  action_step = 0
  last_action = []
  last_action_state = []
  for step in range(TOTAL_STEPS):
    current_simulation_step = step
    """
    if step >= 1000 :
       inject_breakdown("Node1_2_EB_1","Node1_2_EB")
    """
    state = get_state()
    if step % 1 == 0:
      action = get_action_from_policy(compress_state(state))
      apply_action(action)
      action_step = step
      last_action = action
      last_action_state = state
    traci.simulationStep()
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_queue_reward += reward
    if step - action_step == 0:
      update_Q_table(compress_state(last_action_state),last_action, (reward - get_reward(last_action_state))/20.0 ,compress_state(new_state))
    if step % 20 == 0 and step > 0:
      if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Record data every 100 steps
    if step % 100 == 0:
        print(f"\nepsilon :{EPSILON} - SingleAgent_QLearning_Queue.py:313")
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_queue_reward:.2f} - SingleAgent_QLearning_Queue.py:314")
        step_history.append(step)
        reward_history.append(cumulative_queue_reward)
        total_delay = 0
        for edge_id in incoming_edges:
          total_delay += traci.edge.getWaitingTime(edge_id)
        delay_history.append(total_delay)
        queue_history.append(sum(new_state[:-len(tls_ids)]))  # sum of queue lengths
        print(f"Current Qtable length {len(Q_table)}: - SingleAgent_QLearning_Queue.py:322")
  # -------------------------
  # Step 9: Close connection between SUMO and Traci
  # -------------------------
  traci.close()

# Print final Q-table info
print("\nOnline Training completed. Final Qtable size: - SingleAgent_QLearning_Queue.py:329", len(Q_table))
for st, actions in Q_table.items():
    print("State: - SingleAgent_QLearning_Queue.py:331", st, "-> Q-values:", actions)

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
if selected_demand == high :
   plt.title("(high demand)RL Training: Cumulative Reward Queue Length over Steps")
elif selected_demand == medium :
       plt.title("(medium demand)RL Training: Cumulative Reward Queue Length over Steps")
elif selected_demand == low :
       plt.title("(low demand)RL Training: Cumulative Reward Queue Length over Steps")
plt.legend()
plt.grid(True) 
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
if selected_demand == high :
     plt.title("(high demand)RL Training: Queue Length over Steps")
elif selected_demand == medium :
     plt.title("(medium demand)RL Training: Queue Length over Steps")
elif selected_demand == low :
     plt.title("(low demand)RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Vehicle Delay
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Vehicle Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
if selected_demand == high :
    plt.title("(high demand)RL Timing: Total Vehicle Delay over Steps")
elif selected_demand == medium :
    plt.title("(medium demand)RL Timing: Total Vehicle Delay over Steps")
elif selected_demand == low :
    plt.title("(low demand)RL Timing: Total Vehicle Delay over Steps")
plt.legend()
plt.grid(True)
plt.show()

#save results plotted in csv file

data = pd.DataFrame({
    "step": step_history,
    "queue": queue_history,
    "delay": delay_history,
    "cum_queue": reward_history
})
if selected_demand == high :
    data.to_csv("combine graphs/high_demand_SingleAgent_QLearning_Queue_results.csv", index=False)
elif selected_demand == medium :
    data.to_csv("combine graphs/medium_demand_SingleAgent_QLearning_Queue_results.csv", index=False)
elif selected_demand == low :
    data.to_csv("combine graphs/low_demand_SingleAgent_QLearning_Queue_results.csv", index=False)