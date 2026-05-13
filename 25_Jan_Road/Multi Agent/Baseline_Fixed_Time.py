# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization
import pandas as pd # saving results in file 

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
selected_demand = high
rou_files = ["25_Jan_high.rou.xml","25_Jan_medium.rou.xml","25_Jan_low.rou.xml"]
selected_route = rou_files[selected_demand]
Sumo_config = [
    'sumo-gui',
    '-c', '25_Jan.sumocfg',
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
print("TLS IDs: - Baseline_Fixed_Time.py:60", tls_ids) #to depug
print("incoming_edges : - Baseline_Fixed_Time.py:61",incoming_edges) #to depug
print("detectors IDS : - Baseline_Fixed_Time.py:62",all_detectors) #to depug
# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000


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

        print(f"Breakdown injected on vehicle {veh_id} at edge {edge_id} and lane {lane_id} for {duration} steps. - Baseline_Fixed_Time.py:102")
        stopping_car = veh_id

    except traci.TraCIException:
        print("not found - Baseline_Fixed_Time.py:106")
        pass  # Ignore errors if vehicle disappears

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
cumulative_queue_history = []
cumulative_delay_history = []
ttc_history = []
cumulative_reward = 0.0
cumulative_queue_reward = 0.0
cumulative_delay_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning === - Baseline_Fixed_Time.py:146")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    traci.simulationStep()
    state = get_state()
    """
    if step >= 1000 :
       inject_breakdown("Node1_2_EB_1","Node1_2_EB")
    """
    # Calculate total queue and total delay
    total_queue = sum(state[:-len(tls_ids)])
    total_delay = 0
    for edge_id in incoming_edges:
        total_delay += traci.edge.getWaitingTime(edge_id)

    # Update cumulative rewards
    cumulative_delay_reward += -total_delay
    cumulative_queue_reward += -total_queue  # negative queue as reward

    if step % 100 == 0:
        step_history.append(step)
        queue_history.append(total_queue)
        delay_history.append(total_delay)
        cumulative_queue_history.append(cumulative_queue_reward)
        cumulative_delay_history.append(cumulative_delay_reward)
        current_min_ttc = get_network_min_ttc()
        ttc_history.append(current_min_ttc)

        print(f"Step {step}, Total Queue: {total_queue}, Total Delay: {total_delay}, - Baseline_Fixed_Time.py:172"
              f"Cumulative Queue Reward: {cumulative_queue_reward}, "
              f"Cumulative Delay Reward: {cumulative_delay_reward}")

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Queue Reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, cumulative_queue_history, marker='o', linestyle='-', label="Cumulative Queue Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Queue Reward")
if selected_demand == high:
   plt.title("(high demand)Fixed Timing: Cumulative Queue Reward over Steps")
elif selected_demand == medium:
    plt.title("(medium demand)Fixed Timing: Cumulative Queue Reward over Steps")
elif selected_demand == low:
    plt.title("(low demand)Fixed Timing: Cumulative Queue Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Cumulative Delay Reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, cumulative_delay_history, marker='o', linestyle='-', label="Cumulative Delay Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Delay Reward")
if selected_demand == high:
   plt.title("(high demand)Fixed Timing: Cumulative Delay Reward over Steps")
elif selected_demand == medium:
   plt.title("(medium demand)Fixed Timing: Cumulative Delay Reward over Steps")
elif selected_demand == low:
       plt.title("(low demand)Fixed Timing: Cumulative Delay Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
if selected_demand == high:
   plt.title("(high demand)Fixed Timing: Queue Length over Steps")
elif selected_demand == medium:
    plt.title("(medium demand)Fixed Timing: Queue Length over Steps")
elif selected_demand == low:
    plt.title("(low demand)Fixed Timing: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Vehicle Delay
plt.figure(figsize=(10, 6))
plt.plot(step_history, delay_history, marker='o', linestyle='-', label="Total Vehicle Delay")
plt.xlabel("Simulation Step")
plt.ylabel("Total Delay (seconds)")
if selected_demand == high:
   plt.title("(high demand)Fixed Timing: Total Vehicle Delay over Steps")
elif selected_demand == medium:
   plt.title("(medium demand)Fixed Timing: Total Vehicle Delay over Steps") 
elif selected_demand == low:
   plt.title("(low demand)Fixed Timing: Total Vehicle Delay over Steps")
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
    "cum_queue": cumulative_queue_history,
    "cum_delay": cumulative_delay_history
})

if selected_demand == high :
     data.to_csv("combine graphs/high_demand_Baseline_Fixed_result.csv", index=False)
elif selected_demand == medium :
     data.to_csv("combine graphs/medium_demand_Baseline_Fixed_result.csv", index=False)
elif selected_demand == low:
     data.to_csv("combine graphs/low_demand_Baseline_Fixed_result.csv", index=False)



# ==========================================
# TTC Data Export 
# ==========================================

# Save TTC results in a separate CSV file
ttc_data = pd.DataFrame({
    "step": step_history,
    "min_ttc": ttc_history
})

if selected_demand == high:
    ttc_data.to_csv("combine graphs/high_demand_Baseline_Fixed_TTC_results.csv", index=False)
elif selected_demand == medium:
    ttc_data.to_csv("combine graphs/medium_demand_Baseline_Fixed_TTC_results.csv", index=False)
elif selected_demand == low:
    ttc_data.to_csv("combine graphs/low_demand_Baseline_Fixed_TTC_results.csv", index=False)