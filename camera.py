from collections import defaultdict

import cv2
import numpy as np
from cv2 import VideoWriter

from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

real_show = True

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

# Open the video file
video_path = "input_videos/trimmed.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Store the track history
track_history = defaultdict(lambda: [])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (here, MP4)
out = cv2.VideoWriter('output_videos/out.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

len_past = 5
len_to_predict = 3
t = np.arange(len_past)
t_future = np.arange(len_past - 1, len_past + len_to_predict)
past_x_points_future = None
past_y_points_future = None
past_future_points = None
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) > len_past:
                track.pop(0)

            points = np.hstack(track)

            if len(track) == len_past:
                x_points = points.reshape((-1, 2))[:, 0].reshape(-1, 1)
                y_points = points.reshape((-1, 2))[:, 1].reshape(-1, 1)
                lin_reg = LinearRegression()
                lin_reg.fit(t.reshape(-1, 1), x_points)
                x_points_future = lin_reg.predict(t_future.reshape(-1, 1))

                lin_reg = LinearRegression()
                lin_reg.fit(t.reshape(-1, 1), y_points)
                y_points_future = lin_reg.predict(t_future.reshape(-1, 1))

                future_points = np.hstack((x_points_future, y_points_future)).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [future_points], isClosed=False, color=(255, 0, 0), thickness=2)
                cv2.circle(annotated_frame, (int(x_points_future[-1][0]), int(y_points_future[-1][0])), radius=10,
                           color=(255, 0, 0), thickness=5)
                # if past_future_points is not None:
                #     cv2.polylines(annotated_frame, [past_future_points], isClosed=False, color=(255, 255, 0), thickness=2)
                # past_future_points = future_points.copy()
            points = points.astype(np.int32).reshape((-1, 1, 2))

            cv2.circle(annotated_frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=15)
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
