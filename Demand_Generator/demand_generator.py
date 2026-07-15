import cv2
import torch
import numpy as np

# --- PYTORCH 2.6 SECURITY PATCH ---
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_load
# ----------------------------------

from ultralytics import YOLO

# 1. Load the YOLO model
model = YOLO(r"..\NGSIM_YOLO_Model\best.pt")

# 2. Open your video
video_path = r"..\NGSIM_YOLO_Model\dataset\lankershim-camera2-0830am-0845am.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit(f"Error: Could not open video at {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30

# --- POLYGON SETUP ---
poly1 = np.array([[150, 297], [346, 297], [350, 420], [204, 413]], np.int32).reshape(
    (-1, 1, 2)
)
poly2 = np.array([[2, 111], [190, 92], [150, 297], [2, 289]], np.int32).reshape(
    (-1, 1, 2)
)
poly3 = np.array([[198, 2], [320, 2], [348, 89], [190, 92]], np.int32).reshape(
    (-1, 1, 2)
)
poly4 = np.array([[348, 89], [638, 105], [638, 268], [346, 297]], np.int32).reshape(
    (-1, 1, 2)
)
poly5 = np.array([[190, 92], [348, 89], [346, 297], [150, 297]], np.int32).reshape(
    (-1, 1, 2)
)
all_polygons = [poly1, poly2, poly3, poly4, poly5]

polygon_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

# --- EDGE MAPPING FOR SUMO DEMAND ---
# Maps Area ID to (Entry Edge, Exit Edge) based on lankershim.net.xml
AREA_TO_EDGE = {
    1: ("E5", "E4"),  # South (Red)
    2: ("E1", "E2"),  # West (Green)
    3: ("E3", "-E3"),  # North (Blue)
    4: ("-E6", "E6"),  # East (Yellow) - Fixed to 3-lane start edge
}


VALID_ROUTES = {
    "E1": ["E6", "E4"],  # West to East (Straight), West to South (Right)
    "E3": ["E4", "E2", "E6"],  # Added E6: North to East (Left)
    "E5": ["-E3", "E6"],  # South to North (Straight), South to East (Right)
    "-E6": ["E2", "-E3", "E4"],  # Added E4: East to South (Left)
}

# --- TRACKING & QUEUE DETECTION VARIABLES ---
vehicle_demand = {}  # track_id -> {depart, from_edge, to_edge}
vehicle_history = {}  # track_id -> list of (cx, cy)
STOP_THRESHOLD_PIXELS = 5  # Max pixel movement to be considered "stopped"
HISTORY_FRAMES = 15  # How many frames to look back (~0.5 seconds)
current_frame = 0

print("Extracting Origin-Destination demand from video...")
print("Press 'q' to stop early and generate the SUMO route file.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_frame += 1
    annotated_frame = frame.copy()

    # Draw polygons
    for idx, (poly, color) in enumerate(zip(all_polygons, polygon_colors)):
        cv2.polylines(annotated_frame, [poly], isClosed=True, color=color, thickness=2)
        center_x, center_y = int(np.mean(poly[:, 0, 0])), int(np.mean(poly[:, 0, 1]))
        cv2.putText(
            annotated_frame,
            str(idx + 1),
            (center_x - 10, center_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

    # Detect cars
    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml", conf=0.60, verbose=False
    )

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = (
            results[0].boxes.id.cpu().numpy().astype(int)
            if results[0].boxes.id is not None
            else [None] * len(boxes)
        )

        for box, track_id in zip(boxes, track_ids):
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # --- QUEUE (STOPPED CAR) DETECTION LOGIC ---
            if track_id not in vehicle_history:
                vehicle_history[track_id] = []

            vehicle_history[track_id].append((cx, cy))

            if len(vehicle_history[track_id]) > HISTORY_FRAMES:
                vehicle_history[track_id].pop(0)

            is_stopped = False
            if len(vehicle_history[track_id]) == HISTORY_FRAMES:
                old_cx, old_cy = vehicle_history[track_id][0]
                dist_moved = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5
                if dist_moved < STOP_THRESHOLD_PIXELS:
                    is_stopped = True

            # --- DEMAND EXTRACTION LOGIC ---
            area_idx_hit = None
            for area_idx, poly in enumerate(all_polygons, start=1):
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    area_idx_hit = area_idx
                    break

            if area_idx_hit in AREA_TO_EDGE:  # Ignore center intersection (area 5)
                if track_id not in vehicle_demand:
                    # Vehicle spawned! Record depart time and entry edge
                    depart_time = current_frame / fps
                    vehicle_demand[track_id] = {
                        "depart": round(depart_time, 2),
                        "entry_area": area_idx_hit,
                        "exit_area": None,
                    }
                else:
                    # Vehicle already tracked. Check if it reached a new edge
                    if area_idx_hit != vehicle_demand[track_id]["entry_area"]:
                        vehicle_demand[track_id]["exit_area"] = area_idx_hit

            # --- DRAW VISUALS (COMPACT FLOATING CALLOUT) ---
            # 1. Colors: Red (stopped), Cyan (moving)
            box_color = (0, 0, 255) if is_stopped else (255, 255, 0)

            # 2. Draw the Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

            # 3. Create the clean label: "ID:XX (Reg)"
            reg_num = str(area_idx_hit) if area_idx_hit else "N"
            label = f"ID:{track_id} ({reg_num})"

            # Text config: Reduced scale (0.4) for a smaller box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            # Calculate text size
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Position: 5 pixels above bounding box (tighter gap)
            label_y = max(y1 - 5, 20)

            # Draw Background Box (Tighter padding: -3/+3 instead of -5/+5)
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - th - 3),
                (x1 + tw + 4, label_y + 3),
                (0, 0, 0),
                -1,
            )

            # Draw Text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    cv2.imshow("Demand Extraction & Queue Detection", annotated_frame)

    # We use 1ms here because we WANT it to run as fast as possible to process the video quickly
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nUser interrupted. Generating file with collected data...")
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# GENERATE SUMO ROUTE FILE (.rou.xml)
# ==========================================
output_file = r"..\Lankershim Network\extracted_demand.rou.xml"
print(f"\nWriting valid trips to {output_file}...")

with open(output_file, "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write(
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
    )

    # Define the vehicle physics once
    f.write(
        '    <vType id="DEFAULT_VEHTYPE" length="5.0" maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.5"/>\n\n'
    )

    valid_trips = 0
    # Sort vehicles by depart time so SUMO parses them chronologically without errors
    sorted_vehicles = sorted(vehicle_demand.items(), key=lambda item: item[1]["depart"])

    for v_id, data in sorted_vehicles:
        # Only write vehicles where we captured both where they started and where they left
        if data["exit_area"] is not None:
            from_edge = AREA_TO_EDGE[data["entry_area"]][0]
            to_edge = AREA_TO_EDGE[data["exit_area"]][1]

            # --- FILTER: Only write the trip if the physical connection exists ---
            if to_edge in VALID_ROUTES.get(from_edge, []):
                # Use <trip> instead of <vehicle> to allow dynamic routing
                f.write(
                    f'    <trip id="veh_{v_id}" type="DEFAULT_VEHTYPE" depart="{data["depart"]}" from="{from_edge}" to="{to_edge}" departLane="best"/>\n'
                )
                valid_trips += 1
            else:
                # Log dropped trips in the terminal so you know exactly which cars made impossible turns
                print(
                    f"[FILTERED] Dropping veh_{v_id} (No physical route from {from_edge} to {to_edge})"
                )

    f.write("</routes>\n")

print(
    f"\nSUCCESS! Wrote {valid_trips} physically possible trips out of {len(vehicle_demand)} tracked objects."
)
print("You can now load 'extracted_demand.rou.xml' into your SUMO configuration.")
