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
out = cv2.VideoWriter(output_path, fourcc, fps * 2, (frame_width, frame_height))
# ---------------------------------------------

print("Starting Stable ByteTrack Car Detection... Press 'q' to quit.")
print(f"Saving output to: {output_path}")

# 4. Process the video frame by frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video.")
        break

    annotated_frame = frame.copy()

    # 5. Run tracking logic with Class [0] only
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.50,
        imgsz=640,
        verbose=False,
    )

    # 6. Corrected Drawing Logic: Execute if boxes exist, even without IDs
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Safely extract IDs, falling back to None if the tracker hasn't assigned them yet
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [None] * len(boxes)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = track_ids[i]

            # Ground contact point
            cx, cy = (x1 + x2) // 2, y2

            # Always draw the detection box in green
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID: {track_id}" if track_id is not None else "Detecting..."
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # 7. Write the annotated frame to the output video file (this was missing!)
    out.write(annotated_frame)

    # 8. Display the video
    cv2.imshow("ByteTrack Car Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
