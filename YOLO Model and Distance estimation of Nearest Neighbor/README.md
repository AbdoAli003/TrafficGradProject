# YOLO Vehicle Tracking and Distance estimation of Nearest Neighbor 

This folder contains a computer vision module for real-time vehicle detection and tracking. It uses a YOLO object detection model paired with ByteTrack to assign unique tracking IDs to detected vehicles frame-by-frame.

## Prerequisites

Install the required Python dependencies:

```bash
pip install torch numpy opencv-python ultralytics
```
## Setup & Running the Tracker

Since no default video dataset is included, you must update the script to use your own video source.

1. Open `Distance_estimation.py` in a text editor.
2. Locate the `video_path` variable and change it to the path of your own video file:
3. get hemography points of your traffic from specific video frame and from real map top view from Google earth 
4. apply hemography to get near distance estimation of Nearest Neighbor to that in real world 

```python
# For a local video file
video_path = "your_video_file.mp4" 
```
3. Open your terminal, navigate to the `ngsim` folder, and execute the script:

```bash
python Distance_estimation.py
```

**Controls & Output:**

* A video window titled **"ByteTrack Car Tracking"** will open.
* Detected vehicles will have green bounding boxes. Once ByteTrack assigns a trajectory, a unique ID will appear above the box.
* To safely stop the tracking and close the window before the video ends, press **`q`** on your keyboard.