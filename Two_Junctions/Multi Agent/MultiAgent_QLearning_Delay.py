# Step 1: Add modules to provide access to specific libraries and functions
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from traci.constants import CMD_EXECUTEMOVE
import pandas as pd

# Force Python to run inside the exact folder where this script lives ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci

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
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

current_phase = []
stopping_car = None

all_detectors = list(traci.lanearea.getIDList())

incoming_edges = list(set(traci.lane.getEdgeID(lane)
    for tls in traci.trafficlight.getIDList()
    for lane in traci.trafficlight.getControlledLanes(tls)
    if not traci.lane.getEdgeID(lane).startswith(":")
))

tls_ids = list(traci.trafficlight.getIDList())

print("TLS IDs: - MultiAgent_QLearning_Delay.py:61", tls_ids) #to depug
print("incoming_edges: - MultiAgent_QLearning_Delay.py:62",incoming_edges) #to depug
print("all_detectors: - MultiAgent_QLearning_Delay.py:63",all_detectors) #to depug

# RL Hyperparameters
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.992

ACTIONS = [0,1]

# Multi-agent Q-tables
Q_tables = {tls:{} for tls in tls_ids}

MIN_GREEN_STEPS = 40
last_switch_step = {tls:-MIN_GREEN_STEPS for tls in tls_ids}

# detector-edge mapping
detector_edge_map = {}

for det in all_detectors:

    lane_id = traci.lanearea.getLaneID(det)
    edge_id = traci.lane.getEdgeID(lane_id)

    if edge_id.startswith(":"):
        continue

    detector_edge_map[det] = edge_id

edge_list = sorted(list(set(detector_edge_map.values())))

edge_detectors = {edge:[] for edge in edge_list}

for det,edge in detector_edge_map.items():
    edge_detectors[edge].append(det)

print("Edge groups: - MultiAgent_QLearning_Delay.py:102", edge_detectors)

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

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps. - MultiAgent_QLearning_Delay.py:138")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found - MultiAgent_QLearning_Delay.py:142")
        pass  # Ignore errors if vehicle disappears

def get_max_Q_value_of_state(state,tls):

    table = Q_tables[tls]
    if state not in table:
        table[state] = np.zeros(len(ACTIONS))

    return np.max(table[state])

def get_action_from_policy(state,tls):

    table = Q_tables[tls]
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    if state not in table:
        table[state] = np.zeros(len(ACTIONS))

    return int(np.argmax(table[state]))

def update_Q_table(old_state,action,reward,new_state,tls):

    table = Q_tables[tls]
    if old_state not in table:
        table[old_state] = np.zeros(len(ACTIONS))

    if new_state not in table:
        table[new_state] = np.zeros(len(ACTIONS))

    old_q = table[old_state][action]
    best_future = get_max_Q_value_of_state(new_state,tls)
    table[old_state][action] = old_q + ALPHA*(reward + GAMMA*best_future - old_q)

def get_reward():

    total_delay = 0

    for edge in incoming_edges:
        total_delay += traci.edge.getWaitingTime(edge)

    return -total_delay

def get_state():

    queues = []
    for det in all_detectors:
        queues.append(get_queue_length(det))

    phases = []
    for tls in tls_ids:
        phases.append(get_current_phase(tls))

    return tuple(queues + phases)

def compress_state(state):

    num_tls = len(tls_ids)
    queues = state[:-num_tls]
    phases = state[-num_tls:]

    detector_queue = dict(zip(all_detectors,queues))

    edge_bins = []
    for edge in edge_list:
        total = sum(detector_queue[d] for d in edge_detectors[edge])

        if total == 0:
            q_bin = 0
        elif total <= 9:
            q_bin = 1
        elif total <= 18:
            q_bin = 2
        else:
            q_bin = 3

        edge_bins.append(q_bin)

    return tuple(edge_bins + list(phases))

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

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepHaltingNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)
# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

step_history=[]
reward_history=[]
queue_history=[]
delay_history=[]

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning === - MultiAgent_QLearning_Delay.py:252")
print("\n=== Starting Fully Online Continuous Learning === - MultiAgent_QLearning_Delay.py:253")
episodes = 1
for episode in range(episodes):
  if episode !=0:
        if 'SUMO_HOME' in os.environ:
          tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
          sys.path.append(tools)
        else:
          sys.exit("Please declare environment variable 'SUMO_HOME'")
        Sumo_config = [
            'sumo-gui',
             '-c', 'RL.sumocfg',
             '--step-length', '0.10',
             '--delay', '1000',
             '--lateral-resolution', '0']
        traci.start(Sumo_config)
        traci.gui.setSchema("View #0", "real world")
  step_history=[]
  reward_history=[]
  queue_history=[]
  delay_history=[]

  cumulative_reward = 0.0
  EPSILON = 1.0 - episode/episodes
  for step in range(TOTAL_STEPS):

    current_simulation_step = step
    """
    if step >= 1000 :
       inject_breakdown("Node1_2_EB_1","Node1_2_EB")
    """
    state = get_state()
    compressed_state = compress_state(state)

    actions = {}

    # choose action for each agent
    for tls in tls_ids:
        action = get_action_from_policy(compressed_state,tls)
        actions[tls] = action

    # apply actions
    for tls,action in actions.items():
        apply_action(tls,action)

    traci.simulationStep()

    new_state = get_state()
    compressed_new_state = compress_state(new_state)

    reward = get_reward()
    cumulative_reward += reward

    # update each agent
    for tls in tls_ids:
        update_Q_table(compressed_state,actions[tls],reward/100.0,compressed_new_state,tls)

    if step % 20 == 0 and step > 0:
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

    if step % 100 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-len(tls_ids)]))
        delay_history.append(-reward)
        print(f"\nstate:{state} - MultiAgent_QLearning_Delay.py:319")
        print(f"epsilon :{EPSILON} - MultiAgent_QLearning_Delay.py:320")
        print(f"Step {step}, Current_State: {state}, Action: {actions}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f} - MultiAgent_QLearning_Delay.py:321")
        for tls in tls_ids:
           print(f"TLS {tls} Qtable size: {len(Q_tables[tls])} - MultiAgent_QLearning_Delay.py:323")
 # -------------------------
 # Step 9: Close connection between SUMO and Traci
 # -------------------------
  traci.close()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
if selected_demand == high :
    plt.title("(high demand)RL Training: Cumulative delay Reward over Steps")
elif selected_demand == medium:
    plt.title("(medium demand)RL Training: Cumulative delay Reward over Steps")
elif selected_demand == low :
    plt.title("(low demand)RL Training: Cumulative delay Reward over Steps")
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

# Plot Total Delay over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
if selected_demand == high :
       plt.title("(high demand)RL Training: Total Delay over Steps")
elif selected_demand == medium :
       plt.title("(medium demand)RL Training: Total Delay over Steps") 
elif selected_demand == low :
       plt.title("(low demand)RL Training: Total Delay over Steps")
plt.legend()
plt.grid(True)
plt.show()


# Save results

data = pd.DataFrame({
    "step":step_history,
    "queue":queue_history,
    "delay":delay_history,
    "cum_delay":reward_history
})
if selected_demand == high :
    data.to_csv("combine graphs/high_demand_MultiAgent_QLearning_Delay_results.csv",index=False)
elif selected_demand == medium :
    data.to_csv("combine graphs/medium_demand_MultiAgent_QLearning_Delay_results.csv",index=False)
elif selected_demand == low :
    data.to_csv("combine graphs/low_demand_MultiAgent_QLearning_Delay_results.csv",index=False)