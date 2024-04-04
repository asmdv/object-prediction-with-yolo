from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.linear_model import LinearRegression
import json
import pickle
from metrics import MSEWithShift

def create_list_dict():
    return []


class VideoYOLO():
    def __init__(self, model_name, input=None, output=None, past_frame_len=5, prediction_frame_len=1, show=False, debug=False):
        self.model = YOLO(model_name)
        self.show = show
        self.input = input
        self.output = output
        self.past_frame_len = past_frame_len
        self.prediction_frame_len = prediction_frame_len
        self.cap = None
        self.debug = debug
        self.track_history = None
        self.track_predictions = None
    def __read_n_frames(self, n):
        frames = []
        end = False
        for _ in range(n):
            success, frame = self.cap.read()
            if success:
                frames.append(frame)
                pass
            else:
                end = True
                break
        return frames, end

    def loop(self, t, t_future):
        self.track_history = defaultdict(create_list_dict)
        self.track_predictions = defaultdict(create_list_dict)
        # Loop through the video frames
        frame_count = 0
        while self.cap.isOpened():
            # Current frame + prediction frame len
            success, frame = self.cap.read()
            if success:
                results = self.model.track(frame, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                curr_annotated_frame = results[0].plot()
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    x, y, w, h = box
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))

                    track_prediction = self.track_predictions[track_id]

                    points = np.array(track[-self.past_frame_len:])
                    if len(points) == self.past_frame_len:
                        x_points = points.reshape((-1, 2))[:, 0].reshape(-1, 1)
                        y_points = points.reshape((-1, 2))[:, 1].reshape(-1, 1)
                        lin_reg = LinearRegression()
                        lin_reg.fit(t.reshape(-1, 1), x_points)
                        x_points_future = lin_reg.predict(t_future.reshape(-1, 1))

                        lin_reg = LinearRegression()
                        lin_reg.fit(t.reshape(-1, 1), y_points)
                        y_points_future = lin_reg.predict(t_future.reshape(-1, 1))
                        future_points = np.hstack((x_points_future, y_points_future)).astype(np.int32).reshape((-1, 1, 2))
                        track_prediction.append((x_points_future[-1][0], y_points_future[-1][0]))
                        cv2.polylines(curr_annotated_frame, [future_points], isClosed=False, color=(255, 0, 0), thickness=2)
                        cv2.circle(curr_annotated_frame, (int(x_points_future[-1][0]), int(y_points_future[-1][0])), radius=10,
                                   color=(255, 0, 0), thickness=5)
                        # if past_future_points is not None:
                        #     cv2.polylines(annotated_frame, [past_future_points], isClosed=False, color=(255, 255, 0), thickness=2)
                        # past_future_points = future_points.copy()
                    points = points.astype(np.int32).reshape((-1, 1, 2))

                    cv2.circle(curr_annotated_frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=15)
                    cv2.polylines(curr_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                if self.show:
                    cv2.imshow("YOLOv8 Tracking", curr_annotated_frame)
                self.video_writer.write(curr_annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
            frame_count += 1

        with open('track_history.pkl', 'wb') as f:
            pickle.dump(self.track_history, f)
        with open('track_predictions.pkl', 'wb') as f:
            pickle.dump(self.track_predictions, f)

        self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()


    def start(self):
        self.cap = cv2.VideoCapture(self.input)
        assert self.cap.isOpened(), "Error reading video file"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (here, MP4)
        self.video_writer = cv2.VideoWriter(self.output, fourcc, self.cap.get(cv2.CAP_PROP_FPS),
                              (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        t = np.arange(self.past_frame_len)
        t_future = np.arange(self.past_frame_len, self.past_frame_len + self.prediction_frame_len)
        self.loop(t, t_future)

    def eval(self, track_id=None):
        metric = MSEWithShift()
        value = 0
        if track_id:
            value = metric.calc(np.array(self.track_history[track_id]), np.array(self.track_predictions[track_id][:-self.prediction_frame_len]), self.past_frame_len)
        else:
            track_id_len = len(self.track_predictions.keys())
            for track_id in self.track_predictions:
                # Ignore the ones that do not have historical reference.
                # They predicted but do not have past point to compare with.
                if len(np.array(self.track_predictions[track_id][:-self.prediction_frame_len])) == 0:
                    continue
                value += metric.calc(np.array(self.track_history[track_id]), np.array(self.track_predictions[track_id][:-self.prediction_frame_len]), self.past_frame_len)
            value /= track_id_len

        return value
