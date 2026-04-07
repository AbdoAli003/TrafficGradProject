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

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque

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
    'sumo',   # <-- change from 'sumo' to 'sumo-gui'
    '-c', 'RL.sumocfg',
    '--route-files', selected_route,
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]
# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# -------------------------
# Step 6: Define Variables
# -------------------------

current_phase = 0
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
print("TLS IDs: - MultiAgent_DQN_Delay.py:64", tls_ids) #to depug
print("incoming_edges: - MultiAgent_DQN_Delay.py:65",incoming_edges) #to depug
print("detectors_IDS: - MultiAgent_DQN_Delay.py:66",all_detectors) #to depug

TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9992

ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase

NUM_TLS = len(tls_ids)

MIN_GREEN_STEPS = 40
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 200  
# Multi-agent structures
last_switch_step = {tls:-MIN_GREEN_STEPS for tls in tls_ids}
dqn_models = {}
target_models = {}
replay_buffers = {}

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

print("Edge groups: - MultiAgent_DQN_Delay.py:109", edge_detectors)
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

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps. - MultiAgent_DQN_Delay.py:145")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found - MultiAgent_DQN_Delay.py:149")
        pass  # Ignore errors if vehicle disappears

def build_model(state_size, action_size):
    
    """"
    model = keras.Sequential()                                 # Feedforward neural network
    model.add(layers.Input(shape=(state_size,)))               # Input layer
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))  # Output layer
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0005)
    )
    return model
    """
    
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_size, activation = 'linear')
    ])
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0002)
    )

    return model

def to_array(state_tuple):
    state = np.array(state_tuple, dtype=np.float32)
    state[:-NUM_TLS] = np.clip(state[:-NUM_TLS] / 10.0, 0, 1)   # normalize queues (assuming max 50 cars)
    return state.reshape((1, -1))

state_size = len(incoming_edges) + NUM_TLS  # incoming edges + tls per agent
action_size = len(ACTIONS)
# create models per agent
for tls in tls_ids:

    dqn_models[tls] = build_model(state_size, action_size)
    target_models[tls] = build_model(state_size, action_size)
    target_models[tls].set_weights(dqn_models[tls].get_weights())
    replay_buffers[tls] = deque(maxlen=200000)

def get_max_Q_value_of_state(s):
    state_array = to_array(s)
    Q_values = dqn_models(state_array, training=False).numpy()[0]
    return np.max(Q_values)

def get_reward():
    """
    Reward is negative total vehicle delay to minimize delay.
    """
    total_delay = 0.0
    for edge_id in incoming_edges:
        total_delay += traci.edge.getWaitingTime(edge_id)
    return -total_delay

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

def apply_action(actions):

    global last_switch_step, current_simulation_step

    for tls in tls_ids:
        action = actions[tls]

        if current_simulation_step - last_switch_step[tls] < MIN_GREEN_STEPS:
            continue

        if action == 1:
            program = traci.trafficlight.getAllProgramLogics(tls)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls) + 1) % num_phases
            traci.trafficlight.setPhase(tls, next_phase)
            last_switch_step[tls] = current_simulation_step

def update_Q_table(tls, old_state, action, reward, new_state):

    replay_buffers[tls].append((old_state,action,reward,new_state))

    if len(replay_buffers[tls]) < BATCH_SIZE:
        return

    minibatch = random.sample(replay_buffers[tls], BATCH_SIZE)
    states = np.array([to_array(s)[0] for s,_,_,_ in minibatch])
    next_states = np.array([to_array(s2)[0] for _,_,_,s2 in minibatch])

    actions = [a for _,a,_,_ in minibatch]
    rewards = [r for _,_,r,_ in minibatch]
    current_q = dqn_models[tls](states,training=False).numpy()
    next_q_online = dqn_models[tls](next_states,training=False).numpy()
    next_q_target = target_models[tls](next_states,training=False).numpy()
    best_actions = np.argmax(next_q_online,axis=1)
    targets = rewards + GAMMA * next_q_target[np.arange(BATCH_SIZE), best_actions]
    current_q[np.arange(BATCH_SIZE),actions] = targets
    dqn_models[tls].fit(states,current_q,verbose=0)

def get_action_from_policy(state, tls):

    global EPSILON

    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    state_array = to_array(state)
    Q_values = dqn_models[tls](state_array, training=False).numpy()[0]

    return int(np.argmax(Q_values))

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

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepHaltingNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

step_history = []
queue_history = []
delay_history = []
cumulative_delay_history = []
cumulative_reward = 0.0

print("\n=== Starting MultiAgent Online Continuous Learning (DQN, Minimize Delay) === - MultiAgent_DQN_Delay.py:322")
episodes = 1
for episode in range(episodes):
  if episode !=0:
        traci.start(Sumo_config)
  step_history = []
  queue_history = []
  delay_history = []
  cumulative_delay_history = []
  cumulative_reward = 0.0

  for step in range(TOTAL_STEPS):
    current_simulation_step = step
    """
    if step >= 1000 :
       inject_breakdown("Node1_2_EB_1","Node1_2_EB")
    """
    state = get_state()
    compressed_state = compress_state(state)
    if step % 1 == 0:
      actions = {}
      for tls in tls_ids:
        actions[tls] = get_action_from_policy(compressed_state, tls)
      apply_action(actions)

    traci.simulationStep()

    new_state = get_state()
    reward = get_reward()
    cumulative_reward += reward
    if step % 1 == 0:
        for tls in tls_ids:
          update_Q_table(tls,compress_state(state),actions[tls],reward/400.0,compress_state(new_state))
    if step % 20 == 0 and step > 0:
       if EPSILON > EPSILON_MIN:
           EPSILON *= EPSILON_DECAY
    if step % TARGET_UPDATE_FREQ == 0:
        TAU = 0.01
        for tls in tls_ids:
            for target_var, online_var in zip(target_models[tls].variables,dqn_models[tls].variables):
                target_var.assign(TAU * online_var + (1-TAU) * target_var)
    if step % 100 == 0:
        print(f"\nepsilon : {EPSILON} - MultiAgent_DQN_Delay.py:364")
        total_queue = sum(new_state[:-NUM_TLS])
        step_history.append(step)
        delay_history.append(-reward)
        queue_history.append(total_queue)
        cumulative_delay_history.append(cumulative_reward)
        print(f"Step {step}, Total Delay: {reward}, Total Queue: {total_queue}, Cumulative Delay Reward: {cumulative_reward} - MultiAgent_DQN_Delay.py:370")
  # -------------------------
  # Step 9: Close connection between SUMO and Traci
  # -------------------------
  traci.close()

# -------------------------
# Visualization of Results
# -------------------------

# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed. - MultiAgent_DQN_Delay.py:381")
print("DQN Models Summary: - MultiAgent_DQN_Delay.py:382")
for tls in tls_ids:
    print("\nAgent: - MultiAgent_DQN_Delay.py:384",tls)
    dqn_models[tls].summary()

# Plot Cumulative Delay Reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, cumulative_delay_history, marker='o', linestyle='-', label="Cumulative Delay Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Delay Reward")
if selected_demand == high :
      plt.title("(high demand)RL Training (DQN): Cumulative Delay Reward over Steps")
elif selected_demand == medium :
      plt.title("(medium demand)RL Training (DQN): Cumulative Delay Reward over Steps")
elif selected_demand == low :
      plt.title("(low demand)RL Training (DQN): Cumulative Delay Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Vehicle Delay
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Vehicle Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
if selected_demand == high :
      plt.title("(high demand)RL Training (DQN): Total Vehicle Delay over Steps")
elif selected_demand == medium :
      plt.title("(medium demand)RL Training (DQN): Total Vehicle Delay over Steps")
if selected_demand == low :
      plt.title("(low demand)RL Training (DQN): Total Vehicle Delay over Steps") 
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
if selected_demand == high :
    plt.title("(high demand)RL Training (DQN): Total Queue Length over Steps")
elif selected_demand == medium :
    plt.title("(medium demand)RL Training (DQN): Total Queue Length over Steps")
elif selected_demand == low :
    plt.title("(low demand)RL Training (DQN): Total Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

#save results plotted in csv file

data = pd.DataFrame({
    "step": step_history,
    "queue": queue_history,
    "delay": delay_history,
    "cum_delay": cumulative_delay_history
})

if selected_demand == high :
    data.to_csv("combine graphs/high_demand_MultiAgent_DQN_Delay_results.csv", index=False)
elif selected_demand == medium :
    data.to_csv("combine graphs/medium_demand_MultiAgent_DQN_Delay_results.csv", index=False)
elif selected_demand == low :
    data.to_csv("combine graphs/low_demand_MultiAgent_DQN_Delay_results.csv", index=False)