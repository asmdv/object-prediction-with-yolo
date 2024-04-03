import video


video_yolo = video.VideoYOLO("yolov8n.pt", "trimmed.mp4", "out.mp4", show=True)
video_yolo.start()