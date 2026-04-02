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
Sumo_config = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
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
print("TLS IDs: ", tls_ids) #to depug
print("incoming_edges: ",incoming_edges) #to depug
print("all detectors IDs: ",all_detectors) #to depug

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

# Multi-agent Q-tables
Q_tables = {tls:{} for tls in tls_ids}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 120
last_switch_step = {tls:-MIN_GREEN_STEPS for tls in tls_ids}

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

print("Edge groups:", edge_detectors)
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

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps.")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found")
        pass  # Ignore errors if vehicle disappears

def get_max_Q_value_of_state(state,tls):

    table = Q_tables[tls]
    if state not in table:
        table[state] = np.zeros(len(ACTIONS))

    return np.max(table[state])

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

def apply_action(tls,action):

    global current_simulation_step
    if current_simulation_step - last_switch_step[tls] < MIN_GREEN_STEPS:
        return

    if action == 1:
        program = traci.trafficlight.getAllProgramLogics(tls)[0]
        num_phases = len(program.phases)
        next_phase = (traci.trafficlight.getPhase(tls)+1) % num_phases
        traci.trafficlight.setPhase(tls,next_phase)
        last_switch_step[tls] = current_simulation_step

def update_Q_table(old_state,action,reward,new_state,tls):

    table = Q_tables[tls]
    if old_state not in table:
        table[old_state] = np.zeros(len(ACTIONS))

    if new_state not in table:
        table[new_state] = np.zeros(len(ACTIONS))

    old_q = table[old_state][action]
    best_future = get_max_Q_value_of_state(new_state,tls)
    table[old_state][action] = old_q + ALPHA*(reward + GAMMA*best_future - old_q)

def get_action_from_policy(state,tls):
    table = Q_tables[tls]
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    if state not in table:
        table[state] = np.zeros(len(ACTIONS))

    return int(np.argmax(table[state]))

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


print("\n=== Starting Fully Online Continuous Learning ===")
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
  print(f"start of episode {episode}")
  
  last_action_step = 0
  last_action_state = []
  last_actions = {}
  for step in range(TOTAL_STEPS):
    current_simulation_step = step
    """
    if step >= 1000 :
       inject_breakdown("Node1_2_EB_1","Node1_2_EB")
    """
    state = get_state()
    compressed_state = compress_state(state)
    actions = {}
    
    if step % 1 == 0:
      last_action_step = step
      last_action_state = state 
      # choose action for each agent
      for tls in tls_ids:
        action = get_action_from_policy(compressed_state,tls)
        actions[tls] = action

      last_actions = actions
      # apply actions
      for tls,action in actions.items():
        apply_action(tls,action)

    traci.simulationStep()
    new_state = get_state()
    compressed_new_state = compress_state(new_state)
    reward = get_reward(new_state)
    cumulative_queue_reward += reward
    if step - last_action_step == 0:

      # update each agent
      for tls in tls_ids:
        update_Q_table(compress_state(last_action_state),last_actions[tls],(get_reward(last_action_state)-reward)/50.0,compressed_new_state,tls)
    if step % 20 == 0 and step > 0:
      if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Record data every 100 steps
    if step % 100 == 0:
        print(f"\nepsilon :{EPSILON}")
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_queue_reward:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_queue_reward)
        total_delay = 0
        for edge_id in incoming_edges:
          total_delay += traci.edge.getWaitingTime(edge_id)
        delay_history.append(total_delay)
        queue_history.append(sum(new_state[:-len(tls_ids)]))  # sum of queue lengths
        for tls in tls_ids:
           print(f"TLS {tls} Q-table size: {len(Q_tables[tls])}")
  # -------------------------
  # Step 9: Close connection between SUMO and Traci
  # -------------------------
  traci.close()

# Print final Q-table info

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward Queue Length over Steps")
plt.legend()
plt.grid(True) 
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Vehicle Delay
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Vehicle Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
plt.title("RL Timing: Total Vehicle Delay over Steps")
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

data.to_csv("combine graphs/MultiAgent_QLearning_Queue_results.csv", index=False)
