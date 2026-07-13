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
video_path = r"dataset\peachtree-camera8-1245pm-0100pm.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties for the output file
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = "output_tracked.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- UI CONFIGURATIONS ---
BOX_COLOR = (200, 255, 0)     # High-visibility Cyan-Green
TEXT_COLOR = (0, 0, 0)        # Black text for maximum contrast
HUD_BG_COLOR = (20, 20, 20)   # Dark gray/black for the top counter
HUD_TEXT_COLOR = (255, 255, 255) # White text for the counter
# -------------------------

print("Starting Upgraded ByteTrack Car Detection... Press 'q' to quit.")
print(f"Saving output to: {output_path}")

# 4. Process the video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

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
        current_car_count = len(boxes)  # Count vehicles in current frame

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]

            # 1. Draw the main bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # 2. Create a dynamic solid background for the ID text
            label = f"ID: {track_id}" if track_id is not None else "..."
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw the filled rectangle behind the text
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width + 4, y1), BOX_COLOR, -1)
            
            # Draw the text over the filled rectangle
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

    # --- DRAW THE LIVE CAR COUNTER HUD (ULTRA-MINI SCALE) ---
    counter_text = f"Vehicles: {current_car_count}"
    cv2.rectangle(annotated_frame, (10, 10), (105, 34), HUD_BG_COLOR, -1)
    cv2.rectangle(annotated_frame, (10, 10), (105, 34), BOX_COLOR, 1)
    cv2.putText(annotated_frame, counter_text, (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.4, HUD_TEXT_COLOR, 1)
    # --------------------------------------------------------

    # Write the annotated frame
    out.write(annotated_frame)

    # Display the video
    cv2.imshow("ByteTrack Car Tracking", annotated_frame)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        print("Quitting manually...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video successfully saved.")