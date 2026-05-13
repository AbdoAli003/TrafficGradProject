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
rou_files = ["25_Jan_high.rou.xml","25_Jan_medium.rou.xml","25_Jan_low.rou.xml"]
selected_route = rou_files[selected_demand]
Sumo_config = [
    'sumo',   # Use GUI for visualization
    '-c', '25_Jan.sumocfg',
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
print("TLS IDs: - SingleAgent_DQN_Delay.py:65", tls_ids) #to depug
print("incoming_edges: - SingleAgent_DQN_Delay.py:66",incoming_edges) #to depug
print("detectors_IDS: - SingleAgent_DQN_Delay.py:67",all_detectors) #to depug

TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.992

ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase

NUM_TLS = len(tls_ids)

MIN_GREEN_STEPS = 30
last_switch_step = -MIN_GREEN_STEPS
REPLAY_BUFFER = deque(maxlen=50000)
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 200  # update every 500 steps

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

print("Edge groups: - SingleAgent_DQN_Delay.py:107", edge_detectors)
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

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps. - SingleAgent_DQN_Delay.py:143")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found - SingleAgent_DQN_Delay.py:147")
        pass  # Ignore errors if vehicle disappears

def build_model(state_size, action_size):
    
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
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation = 'linear')
    ])
    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0002)
    )

    return model
    """

def to_array(state_tuple):
    state = np.array(state_tuple, dtype=np.float32)
    state[:-NUM_TLS] = np.clip(state[:-NUM_TLS] / 15.0, 0, 1)   # normalize queues (assuming max 50 cars)
    return state.reshape((1, -1))

state_size = len(incoming_edges) + NUM_TLS  # incoming edges + tlses
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)
target_model = build_model(state_size, action_size)
target_model.set_weights(dqn_model.get_weights())

def get_max_Q_value_of_state(s):
    state_array = to_array(s)
    Q_values = dqn_model(state_array, training=False).numpy()[0]
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

def apply_action(action):
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

def update_Q_table(old_state, action, reward, new_state):
    # Store experience
    REPLAY_BUFFER.append((old_state, action, reward, new_state))

    if len(REPLAY_BUFFER) < BATCH_SIZE:
        return

    minibatch = random.sample(REPLAY_BUFFER, BATCH_SIZE)

    states = np.array([to_array(s)[0] for s, _, _, _ in minibatch])
    next_states = np.array([to_array(s_next)[0] for _, _, _, s_next in minibatch])

    actions = [a for _, a, _, _ in minibatch]
    rewards = [r for _, _, r, _ in minibatch]

    # Predict in batch
    current_q = dqn_model(states, training=False).numpy()
    next_q_online = dqn_model(next_states, training=False).numpy()
    next_q_target = target_model(next_states, training=False).numpy()

    best_actions = np.argmax(next_q_online, axis=1)

    targets = rewards + GAMMA * next_q_target[np.arange(BATCH_SIZE), best_actions]

    current_q[np.arange(BATCH_SIZE), actions] = targets


    dqn_model.fit(states, current_q, verbose=0)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model(state_array, training=False).numpy()[0]
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


def get_network_min_ttc(safe_threshold=50.0):
    """
    Calculates the minimum Time To Collision (TTC) across all vehicles currently in the network.
    Returns safe_threshold if no vehicles are on a collision course.
    """
    min_ttc = safe_threshold
    vehicles = traci.vehicle.getIDList()
    
    for veh_id in vehicles:
        # getLeader returns a tuple (leader_id, distance) or None
        leader_info = traci.vehicle.getLeader(veh_id, 0.0) 
        
        if leader_info is not None:
            leader_id, distance = leader_info
            v_follower = traci.vehicle.getSpeed(veh_id)
            v_leader = traci.vehicle.getSpeed(leader_id)
            
            # TTC is only valid if the follower is faster than the leader
            if v_follower > v_leader:
                relative_speed = v_follower - v_leader
                # Prevent division by zero just in case
                if relative_speed > 0: 
                    ttc = distance / relative_speed
                    if ttc < min_ttc:
                        min_ttc = ttc
                        
    return min_ttc    

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

step_history = []
queue_history = []
delay_history = []
cumulative_delay_history = []
ttc_history = []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN, Minimize Delay) === - SingleAgent_DQN_Delay.py:325")
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
    if step % 1 == 0:
      action = get_action_from_policy(compress_state(state))
      apply_action(action)

    traci.simulationStep()

    new_state = get_state()
    reward = get_reward()
    cumulative_reward += reward
    if step % 1 == 0:
       update_Q_table(compress_state(state), action, reward/100.0, compress_state(new_state))
    if step % 20 == 0 and step > 0:
       if EPSILON > EPSILON_MIN:
           EPSILON *= EPSILON_DECAY
    if step % TARGET_UPDATE_FREQ == 0:
       TAU = 0.01
       for target_var, online_var in zip(target_model.variables, dqn_model.variables):
           target_var.assign(TAU * online_var + (1 - TAU) * target_var)
    if step % 100 == 0:
        print(f"\nepsilon : {EPSILON} - SingleAgent_DQN_Delay.py:362")
        total_queue = sum(new_state[:-NUM_TLS])
        step_history.append(step)
        delay_history.append(-reward)
        queue_history.append(total_queue)
        cumulative_delay_history.append(cumulative_reward)
        current_min_ttc = get_network_min_ttc()
        ttc_history.append(current_min_ttc)
        print(f"Step {step}, Total Delay: {reward}, Total Queue: {total_queue}, Cumulative Delay Reward: {cumulative_reward} - SingleAgent_DQN_Delay.py:368")
  # -------------------------
  # Step 9: Close connection between SUMO and Traci
  # -------------------------
  traci.close()

# -------------------------
# Visualization of Results
# -------------------------

# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed. - SingleAgent_DQN_Delay.py:379")
print("DQN Model Summary: - SingleAgent_DQN_Delay.py:380")
dqn_model.summary()

# Plot Cumulative Delay Reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, cumulative_delay_history, marker='o', linestyle='-', label="Cumulative Delay Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Delay Reward")
if selected_demand == high:
   plt.title("(high demand)RL Training (DQN): Cumulative Delay Reward over Steps")
elif selected_demand == medium:
   plt.title("(medium demand)RL Training (DQN): Cumulative Delay Reward over Steps")
elif selected_demand == low:
   plt.title("(low demand)RL Training (DQN): Cumulative Delay Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Vehicle Delay
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Vehicle Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
if selected_demand == high:
    plt.title("(high demand)RL Training (DQN): Total Vehicle Delay over Steps")
elif selected_demand == medium:
    plt.title("(medium demand)RL Training (DQN): Total Vehicle Delay over Steps")
elif selected_demand == low:
    plt.title("(low demand)RL Training (DQN): Total Vehicle Delay over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
if selected_demand == high:
    plt.title("(high demand)RL Training (DQN): Total Queue Length over Steps")
elif selected_demand == medium:
    plt.title("(medium demand)RL Training (DQN): Total Queue Length over Steps")
elif selected_demand == low:
    plt.title("(low demand)RL Training (DQN): Total Queue Length over Steps")

plt.legend()
plt.grid(True)
plt.show()

# Plot Minimum TTC over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, ttc_history, marker='o', linestyle='-', color='red', label="Network Min TTC")
plt.xlabel("Simulation Step")
plt.ylabel("Minimum Time to Collision (Seconds)")
plt.axhline(y=3.0, color='orange', linestyle='--', label='Critical Threshold (3s)') # Optional: Reference line for danger

if selected_demand == high:
    plt.title("(high demand) RL Safety: Min TTC over Steps")
elif selected_demand == medium:
    plt.title("(medium demand) RL Safety: Min TTC over Steps")
elif selected_demand == low:
    plt.title("(low demand) RL Safety: Min TTC over Steps")
    
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
    data.to_csv("combine graphs/high_demand_SingleAgent_DQN_Delay_results.csv", index=False)
elif selected_demand == medium :
    data.to_csv("combine graphs/medium_demand_SingleAgent_DQN_Delay_results.csv", index=False)
elif selected_demand == low :
    data.to_csv("combine graphs/low_demand_SingleAgent_DQN_Delay_results.csv", index=False)


# ==========================================
# TTC Data Export 
# ==========================================

# Save TTC results in a separate CSV file
ttc_data = pd.DataFrame({
    "step": step_history,
    "min_ttc": ttc_history
})

if selected_demand == high:
    ttc_data.to_csv("combine graphs/high_demand_SingleAgent_DQN_Delay_TTC_results.csv", index=False)
elif selected_demand == medium:
    ttc_data.to_csv("combine graphs/medium_demand_SingleAgent_DQN_Delay_TTC_results.csv", index=False)
elif selected_demand == low:
    ttc_data.to_csv("combine graphs/low_demand_SingleAgent_DQN_Delay_TTC_results.csv", index=False)