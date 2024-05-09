import mlflow
import string
import random
def generate_random_string(length):
    characters = string.ascii_letters + string.digits

    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string




def main():
    tracking_uri = " http://127.0.0.1:8090"
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("yolo")

    predictor_type = dict()

    import json
    import os

    folder_path = "/Users/asif/progs/02-uni/05-advanced-proj-wang/custom-yolo/experiment"

    file_list = []

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            with open(os.path.join(folder_path, file_name), "r") as json_file:
                data = json.load(json_file)
                random_string = generate_random_string(6)

            with mlflow.start_run(run_name=f"{data['predictor_name']}_{data['past_len']}_{data['future_len']}_{random_string}"):
                mlflow.log_params(data)

                mlflow.log_metric("mse_avg", data['mse_avg'])
                mlflow.set_tag("predictor name", data['predictor_name'])
                if data['predictor_name'] in predictor_type:
                    predictor_type[data['predictor_name']].append(data['mse_avg'])
                else:
                    predictor_type[data['predictor_name']] = [data['mse_avg']]
    for type in predictor_type:
        with mlflow.start_run(run_name=f"{type}_avg"):
            mlflow.log_metric("mse_avg", sum(predictor_type[type]) / len(predictor_type[type]))
            mlflow.set_tag("predictor name", type)

if __name__ == "__main__":
    main()