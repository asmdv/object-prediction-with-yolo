import video


video_yolo = video.VideoYOLO("yolov8n.pt", "input_videos/trimmed.mp4", "output_videos/out.mp4", past_frame_len=5, prediction_frame_len=1, show=True, debug=True)
video_yolo.start()
print(video_yolo.eval())
