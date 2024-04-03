from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import cv2
from sklearn.linear_model import LinearRegression

class VideoYOLO():
    def __init__(self, model_name, input=None, output=None, past_frame_len=5, prediction_frame_len=1, show=False):
        self.model = YOLO(model_name)
        self.show = show
        self.input = input
        self.output = output
        self.past_frame_len = past_frame_len
        self.prediction_frame_len = prediction_frame_len
        self.cap = None

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
        track_history = defaultdict(lambda: [])
        # Loop through the video frames
        while self.cap.isOpened():
            # Read a frame from the video
            frames, end = self.__read_n_frames(self.prediction_frame_len)
            for frame in frames:
                results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                curr_annotated_frame = results[0].plot()
                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))

                    if len(track) > self.past_frame_len:
                        track.pop(0)

                    points = np.hstack(track)

                    if len(track) == self.past_frame_len:
                        x_points = points.reshape((-1, 2))[:, 0].reshape(-1, 1)
                        y_points = points.reshape((-1, 2))[:, 1].reshape(-1, 1)
                        lin_reg = LinearRegression()
                        lin_reg.fit(t.reshape(-1, 1), x_points)
                        x_points_future = lin_reg.predict(t_future.reshape(-1, 1))

                        lin_reg = LinearRegression()
                        lin_reg.fit(t.reshape(-1, 1), y_points)
                        y_points_future = lin_reg.predict(t_future.reshape(-1, 1))

                        future_points = np.hstack((x_points_future, y_points_future)).astype(np.int32).reshape((-1, 1, 2))
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
            if end:
                break

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
        t_future = np.arange(self.past_frame_len - 1, self.past_frame_len + self.prediction_frame_len)
        self.loop(t, t_future)

