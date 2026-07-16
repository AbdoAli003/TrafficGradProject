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

            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 1)
            cv2.circle(annotated_frame, (car_center_x, car_center_y), 3, BOX_COLOR, -1)

            
            label = f"ID: {track_id}" if track_id is not None else "Vehicle"
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