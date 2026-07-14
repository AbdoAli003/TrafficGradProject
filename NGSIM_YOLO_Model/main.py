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
model = YOLO("best.pt")

# 2. Open your video
video_path = r"dataset\lankershim-camera2-0830am-0845am.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- DYNAMIC TIME TRACKER ---
current_frame = 0
# ----------------------------

# Define the codec and create VideoWriter object
output_path = "output_tracked.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- UI CONFIGURATIONS ---
BOX_COLOR = (200, 255, 0)        # High-visibility Cyan-Green
TEXT_COLOR = (0, 0, 0)           # Black text for maximum contrast
HUD_BG_COLOR = (20, 20, 20)      # Dark gray/black for the top counter
HUD_TEXT_COLOR = (255, 255, 255) # White text for the counter
# -------------------------

# --- POLYGON SETUP ---
poly1 = np.array([[150, 297], [346, 297], [350, 420], [204, 413]], np.int32).reshape((-1, 1, 2))
poly2 = np.array([[2, 111], [190, 92], [150, 297], [2, 289]], np.int32).reshape((-1, 1, 2))
poly3 = np.array([[198, 2], [320, 2], [348, 89], [190, 92]], np.int32).reshape((-1, 1, 2))
poly4 = np.array([[348, 89], [638, 105], [638, 268], [346, 297]], np.int32).reshape((-1, 1, 2))
poly5 = np.array([[190, 92], [348, 89], [346, 297], [150, 297]], np.int32).reshape((-1, 1, 2))

all_polygons = [poly1, poly2, poly3, poly4, poly5]

polygon_colors = [
    (0, 0, 255),    # Red for poly1
    (0, 255, 0),    # Green for poly2
    (255, 0, 0),    # Blue for poly3
    (0, 255, 255),  # Yellow for poly4
    (255, 0, 255)   # Magenta for poly5
]
# ---------------------

# --- SUMO DEMAND TRACKER ---
unique_cars_per_area = {1: set(), 2: set(), 3: set(), 4: set(), 5: set()}
# ---------------------------

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"📍 Clicked coordinates: (X: {x}, Y: {y})")

print("Starting Upgraded ByteTrack Car Detection... Press 'q' to quit at any time.")
print(f"Saving output to: {output_path}")

cv2.namedWindow("ByteTrack Car Tracking")
cv2.setMouseCallback("ByteTrack Car Tracking", click_event)

# 4. Process the video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video reached.")
        break
        
    current_frame += 1

    annotated_frame = frame.copy()

    # --- DRAW POLYGONS & THEIR NUMBERS ---
    for idx, (poly, color) in enumerate(zip(all_polygons, polygon_colors)):
        cv2.polylines(annotated_frame, [poly], isClosed=True, color=color, thickness=2)
        center_x = int(np.mean(poly[:, 0, 0]))
        center_y = int(np.mean(poly[:, 0, 1]))
        cv2.putText(annotated_frame, str(idx + 1), (center_x - 10, center_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    # -------------------------------------

    # Run tracking logic
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.60,
        verbose=False,
    )

    current_car_count = 0

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        current_car_count = len(boxes)  

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]

            car_center_x = (x1 + x2) // 2
            car_center_y = (y1 + y2) // 2

            area_label = "-"
            for area_idx, poly in enumerate(all_polygons):
                if cv2.pointPolygonTest(poly, (car_center_x, car_center_y), False) >= 0:
                    area_label = str(area_idx + 1)
                    if track_id is not None:
                        unique_cars_per_area[area_idx + 1].add(track_id)
                    break 

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cv2.circle(annotated_frame, (car_center_x, car_center_y), 3, BOX_COLOR, -1)

            label = f"A{area_label} | ID: {track_id}" if track_id is not None else f"A{area_label}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), BOX_COLOR, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

    # --- DRAW THE LIVE CAR COUNTER HUD (ULTRA-MINI SCALE) ---
    elapsed_time_sec = current_frame // fps if fps > 0 else 0
    counter_text = f"Vehicles: {current_car_count} | Elapsed: {elapsed_time_sec}s"
    cv2.rectangle(annotated_frame, (10, 10), (250, 34), HUD_BG_COLOR, -1)
    cv2.rectangle(annotated_frame, (10, 10), (250, 34), BOX_COLOR, 1)
    cv2.putText(annotated_frame, counter_text, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.4, HUD_TEXT_COLOR, 1)
    # --------------------------------------------------------

    out.write(annotated_frame)
    cv2.imshow("ByteTrack Car Tracking", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\n[INFO] 'q' pressed. Quitting manually...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video successfully saved.")

# --- AUTO-GENERATE SUMO ROUTES FILE ---
final_elapsed_seconds = current_frame // fps if fps > 0 else 0

edge_mapping = {
    1: {"from": "-405366771#1", "to": "-405366764#1"},
    2: {"from": "405366764#0", "to": "405366771#0"},
    3: {"from": "508815228#0", "to": "518179568#0"},
    4: {"from": "518179569#0", "to": "508815234#0"},
    5: {"from": "", "to": ""}
}

xml_content = "<routes>\n"
xml_content += '    <vType id="standard_car" vClass="passenger" maxSpeed="27.78" accel="2.6" decel="4.5" length="5.0"/>\n\n'

for area, car_ids in unique_cars_per_area.items():
    count = len(car_ids)
    from_edge = edge_mapping[area]["from"]
    to_edge = edge_mapping[area]["to"]
    
    if count > 0 and from_edge != "":  
        xml_content += f'    <flow id="Area{area}_Traffic" type="standard_car" begin="0" end="{final_elapsed_seconds}" number="{count}" from="{from_edge}" to="{to_edge}"/>\n'

xml_content += "</routes>"

with open("video_demand.rou.xml", "w") as f:
    f.write(xml_content)

print(f"\n[SUCCESS] Automatically generated 'video_demand.rou.xml' for {final_elapsed_seconds} seconds of traffic!")