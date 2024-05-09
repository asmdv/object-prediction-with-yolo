import pickle
import torch
import numpy as np
from src.predictorinterface import LinearRegressionPredictor, KalmanFilterPredictor, LSTMPredictor
import custom_metrics
import json
import time

def create_list_dict():
    return []
def dict_to_tuple(dict):
    l = [v for k, v in dict.items()]
    return tuple(l)

def calculate_avg(track_history_path, past_len, future_len, predictor, metric):
    with open(track_history_path, 'rb') as file:
        track_history = pickle.load(file)
    ms_errors = []
    for key in track_history.keys():
        track_past = torch.stack(track_history[key])
        track_predictions = []
        i = 0
        while i < len(track_past) - past_len:
            out = predictor.predict(track_past[i:i + past_len], future_len)
            track_predictions.append(out[-1][-1])
            i += 1
        if track_predictions:
            track_predictions = np.array(track_predictions)
            if len(track_predictions[:-past_len]):
                value = metric.calc(track_past,
                                    track_predictions[:-past_len], past_len)
                ms_errors.append(value)
    return sum(ms_errors) / len(ms_errors)

class Experiment:
    def __init__(self, past_len, future_len, predictor):
        self.past_len = past_len
        self.future_len = future_len
        self.predictor = predictor
        self.mse_avg = None
        self.exec_time = None

    def get_info(self):
        return dict(past_len=self.past_len, future_len=self.future_len, **self.predictor.get_info(), exec_time=self.exec_time, mse_avg=self.mse_avg)
    def __str__(self):
        d = self.get_info()
        s = f'{d["predictor_name"]}'
        for k, v in d.items():
            if k != "predictor_name":
                s += f'_{v}'
        # s += f'_{d["mse_avg"]}'

        return s
    def set_mse_avg(self, mse_avg):
        self.mse_avg = round(mse_avg, 3)

    def set_exec_time(self, exec_time):
        self.exec_time = exec_time
def read_configuration(file_path):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)

    for i in range(len(config)):
        predictor_obj = eval(config[i]["predictor"]["name"])
        config[i]["predictor"] = predictor_obj(**config[i]["predictor"]["params"])
    for i in range(len(config)):
        print(config[i])
    return config
def main(config_path):
    config = read_configuration(config_path)
    experiment = Experiment(3, 3, LinearRegressionPredictor())
    for c in config:
        experiment = Experiment(**c)
        print(f"Experiment: {experiment}, Time: ", end="")
        t = time.time()
        mse_avg = float(calculate_avg(
            'track_history_amsterdam-short.mp4.pkl',
            past_len=experiment.past_len,
            future_len=experiment.future_len,
            predictor=experiment.predictor,
            metric=custom_metrics.MSEWithShift()
        ))
        print(f"{time.time() - t:.3}s")
        experiment.set_mse_avg(mse_avg)
        experiment.set_exec_time(round(time.time() - t, 2))
        file_path = f"experiment/{experiment}.json"
        json_string = json.dumps(experiment.get_info(), indent=4)
        with open(file_path, "w") as json_file:
            json_file.write(json_string)

if __name__ == "__main__":
    config_paths = ["configuration-linear-regression-2-2.10.json"]
    for config_path in config_paths:
        config_path = f"/Users/asif/progs/02-uni/05-advanced-proj-wang/custom-yolo/config/{config_path}"
        main(config_path)
