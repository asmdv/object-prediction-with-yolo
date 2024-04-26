from src import video
from src.predictorinterface import LinearRegressionPredictor, KalmanFilterPredictor, LSTMPredictor


video_yolo = video.VideoYOLO("yolov8n.pt", "input_videos/Stockholm-Walks-Luntmakargatan-short-1.mp4", "output_videos/out.mp4", past_frame_len=10, prediction_frame_len=3, show=True, debug=True, predictor=LSTMPredictor())
video_yolo.start()
print(video_yolo.eval())
