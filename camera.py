from collections import defaultdict

import cv2
import numpy as np
from cv2 import VideoWriter

from ultralytics import YOLO
from ultralytics.solutions import speed_estimation


real_show = True

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "trimmed.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Store the track history
track_history = defaultdict(lambda: [])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (here, MP4)
out = cv2.VideoWriter('out.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.circle(annotated_frame, (int(x),int(y)), radius=0, color=(0,0,255), thickness=15)
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        if real_show:
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()