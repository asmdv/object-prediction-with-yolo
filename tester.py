from src import video
import argparse
from src.predictorinterface import LinearRegressionPredictor, KalmanFilterPredictor, LSTMPredictor


def main(args):
    video_yolo = video.VideoYOLO(args.model, "input_videos/stockholm-demo.mp4", "output_videos/out.mp4", past_frame_len=3, prediction_frame_len=3, show=True, debug=True, predictor=LSTMPredictor())
    video_yolo.start()
    print(video_yolo.eval())



def main(argument1, argument2):
    # Your main function logic here
    print("Argument 1:", argument1)
    print("Argument 2:", argument2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Description of argument 1")
    parser.add_argument("--input_video", type=str, required=True)

    args = parser.parse_args()

    main(args)
