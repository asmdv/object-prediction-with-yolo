import cv2
input = "input_videos/trimmed.mp4"
output = "output_videos/output_track_comparison.mp4"
cap = cv2.VideoCapture(input)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (here, MP4)
video_writer = cv2.VideoWriter(output, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

past_frame_len = 5
prediction_frame_len = 1

import pickle

# Load dictionary from the pickle file
with open('track_history_old.pkl', 'rb') as f:
    track_history = pickle.load(f)

with open('track_predictions_old.pkl', 'rb') as f:
    track_predictions = pickle.load(f)

frame_count = 0
while cap.isOpened():
    # Current frame + prediction frame len
    success, frame = cap.read()
    if success:
        # results = model.track(frame, persist=True)
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()
        # curr_annotated_frame = results[0].plot()
        # for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
        #     x, y, w, h = box
        #     track = self.track_history[track_id]
        #     track.append((float(x), float(y)))
        #
        #     track_prediction = self.track_predictions[track_id]
        #
        #     points = np.array(track[-self.past_frame_len:])
        #     if len(points) == self.past_frame_len:
        #         x_points = points.reshape((-1, 2))[:, 0].reshape(-1, 1)
        #         y_points = points.reshape((-1, 2))[:, 1].reshape(-1, 1)
        #         lin_reg = LinearRegression()
        #         lin_reg.fit(t.reshape(-1, 1), x_points)
        #         x_points_future = lin_reg.predict(t_future.reshape(-1, 1))
        #
        #         lin_reg = LinearRegression()
        #         lin_reg.fit(t.reshape(-1, 1), y_points)
        #         y_points_future = lin_reg.predict(t_future.reshape(-1, 1))
        #         future_points = np.hstack((x_points_future, y_points_future)).astype(np.int32).reshape((-1, 1, 2))
        #         track_prediction.append((x_points_future[-1][0], y_points_future[-1][0]))
        #         cv2.polylines(curr_annotated_frame, [future_points], isClosed=False, color=(255, 0, 0), thickness=2)
        #         cv2.circle(curr_annotated_frame, (int(x_points_future[-1][0]), int(y_points_future[-1][0])), radius=10,
        #                    color=(255, 0, 0), thickness=5)
        #     points = points.astype(np.int32).reshape((-1, 1, 2))

        #     cv2.circle(curr_annotated_frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=15)
        #     cv2.polylines(curr_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        #
        # if self.show:
        #     cv2.imshow("YOLOv8 Tracking", curr_annotated_frame)
        if frame_count < past_frame_len:
            frame_count += 1
            continue
        cv2.circle(frame, (int(track_history[8][frame_count][0]), int(track_history[8][frame_count][1])), radius=0,
                   color=(0, 0, 255), thickness=15)
        cv2.circle(frame, (int(track_predictions[8][frame_count-past_frame_len][0]), int(track_history[8][frame_count-past_frame_len][1])), radius=0, color=(255,0,0), thickness=15)
        cv2.imshow("", frame)
        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    frame_count += 1

video_writer.release()
cap.release()
cv2.destroyAllWindows()