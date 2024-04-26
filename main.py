from src import video
from src.predictorinterface import LinearRegressionPredictor, KalmanFilterPredictor


video_yolo = video.VideoYOLO("yolov8n.pt", "input_videos/Stockholm-Walks-Luntmakargatan-short-1.mp4", "output_videos/out.mp4", past_frame_len=10, prediction_frame_len=10, show=True, debug=True, predictor=LinearRegressionPredictor())
video_yolo.start()
print(video_yolo.eval())
