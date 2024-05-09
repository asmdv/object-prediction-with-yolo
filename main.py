from src import video
from src.predictorinterface import LinearRegressionPredictor, KalmanFilterPredictor, LSTMPredictor
import argparse


def main(args):
    video_yolo = video.VideoYOLO("yolov8l.pt", "input_videos/amsterdam-full.mp4", "output_videos/amsterdam.mp4", past_frame_len=20, prediction_frame_len=3, show=True, debug=True, predictor=None)
    video_yolo.start()
    print(video_yolo.eval())


if __name__ == "__main__":
    main(None)
