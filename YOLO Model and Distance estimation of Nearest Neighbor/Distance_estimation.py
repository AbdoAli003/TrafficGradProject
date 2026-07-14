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

video_path = r"../Videos of Graduation project\lankershim 08-30am_08-45am\8_30am_08_45am\lankershim-camera2-0830am-0845am.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path} - main.py:28")
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

print("Starting Upgraded ByteTrack Car Detection... Press 'q' to quit. - main.py:48")
print(f"Saving output to: {output_path} - main.py:49")

image_points = np.float32([
    [347,84],
    [348,265],
    [142,265],
    [144,106]
])

world_points = np.float32([
    [32,39],
    [0,39],
    [0,0],
    [30,0]
])

H, _ = cv2.findHomography(image_points, world_points)

# 4. Process the video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    cars = []
    if not success:
        print("End of video. - main.py:72")
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
            
            cx = (x1 + x2) // 2
            cy = y2
            point = np.array([[[cx, cy]]], dtype=np.float32)

            world = cv2.perspectiveTransform(point, H)

            X = world[0][0][0]
            Y = world[0][0][1]
            cars.append({
                "id": track_id,
                "cx": cx,
                "cy": cy,
                "X": X,
                "Y": Y,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
            # 1. Draw the main bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # 2. Create a dynamic solid background for the ID text
            label = f"{track_id}" if track_id is not None else "..."
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.3
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            pad = 1

            badge_w = text_w + pad * 1
            badge_h = text_h + pad * 1

            # Center of bounding box
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2

            badge_x1 = cx_box - badge_w // 2
            badge_y1 = cy_box - badge_h // 2
            badge_x2 = badge_x1 + badge_w
            badge_y2 = badge_y1 + badge_h
            
            # Shadow / outline
            cv2.putText(
                annotated_frame,
                label,
                (badge_x1 + pad, badge_y2 - pad),
                font,
                font_scale,
                (0, 0, 0),      # Black
                2,              # Thick outline
                cv2.LINE_AA
            )
            # White ID
            cv2.putText(
                annotated_frame,
                label,
                (badge_x1 + pad,badge_y2 - pad),
                font,
                font_scale,
                (255,255,255),
                thickness,
                cv2.LINE_AA
            )
    for i in range(len(cars)):
    
        nearest = None
        nearest_distance = float("inf")

        for j in range(len(cars)):

            if i == j:
                continue

            dx = cars[i]["X"] - cars[j]["X"]
            dy = cars[i]["Y"] - cars[j]["Y"]

            d = np.sqrt(dx*dx + dy*dy)

            if d < nearest_distance:
                nearest_distance = d
                nearest = cars[j]

        if nearest is None:
            continue
        
        nearest_distance = nearest_distance * 0.5
        
        if nearest_distance < 2:
            color = (0,0,255)
        elif nearest_distance < 5:
            color = (0,165,255)

        elif nearest_distance < 10:
            color = (0,255,255)

        else:
            color = (0,255,0)
        
        cv2.line(
            annotated_frame,
            (cars[i]["cx"], cars[i]["cy"]),
            (nearest["cx"], nearest["cy"]),
            color,
            2
        )

        mx = (cars[i]["cx"] + nearest["cx"]) // 2
        my = (cars[i]["cy"] + nearest["cy"]) // 2
        nearest_id = nearest["id"]
        info = f"N:{nearest_id} | {nearest_distance:.1f} m"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(info, font, font_scale, thickness)

        pad = 1

        # Top-left corner of the label
        label_x = cars[i]["x1"]
        label_y = cars[i]["y1"] - 2

        # Filled background
        cv2.rectangle(
            annotated_frame,
            (label_x, label_y - text_h - pad),
            (label_x + text_w + 2 * pad, label_y + pad),
            (40, 40, 40),
            -1
        )

        # Small colored strip indicating danger level
        cv2.rectangle(
            annotated_frame,
            (label_x, label_y - text_h - pad),
            (label_x + 4, label_y + pad),
            color,
            -1
        )
        # Draw text
        cv2.putText(
            annotated_frame,
            info,
            (label_x + 8, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
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
        print("Quitting manually... - main.py:274")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video successfully saved. - main.py:280")