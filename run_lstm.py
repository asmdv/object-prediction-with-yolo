import pickle
import argparse
import os
import src.rnn as rnn
import datetime
import sys
import src.predictorinterface as predictorinterface


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as err:
            print(f"Error: Creating directory '{directory_path}' - {err}")
    else:
        print(f"Directory '{directory_path}' already exists.")

def create_experiment_path(args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H_%M_%S")
    experiment_name = f"finetuning/exp_{args.model}_{args.num_layers}_{args.hidden_dim}_epochs_{args.epochs}_{args.lr}_{formatted_time}"
    create_directory_if_not_exists(experiment_name)
    create_directory_if_not_exists(f"{experiment_name}/plots")
    return experiment_name


def Merge(dict1, dict2):
    return(dict1.update(dict2))
def create_list_dict():
    return []

def change_keys(dictionary, key_transform_func):
    return {key_transform_func(key): value for key, value in dictionary.items()}

def add_file_variable_name(key, i):
    return tuple(list(key) + [i])


def main(args):
    import torch

    experiment_name = create_experiment_path(args)

    f = open(f'{experiment_name}/log.txt', 'a')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    print(args)
    track_history_paths = [f"pickle/track_history_amsterdam_full_000{i}.mp4.pkl" for i in range(7)]
    track_histories = []
    track_histories_sum = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for counter, track_history_path in enumerate(track_history_paths):
        print(f"Loading {track_history_path}")
        with open(track_history_path, 'rb') as file:
            dict_with_changed_keys = change_keys(pickle.load(file), lambda x: add_file_variable_name(x, counter))
            track_histories.append(dict_with_changed_keys)

    for track_history in track_histories:
        Merge(track_histories_sum, track_history)


    test_track_histories = []
    test_track_keys = [(2470, 'truck', 6),
                       (2410, 'person', 6),
                       (555, 'motorcycle', 6),
                       (492, 'car', 6),
                       (85, 'person', 6),
                       (3690, 'car', 5),
                       (3539, 'person', 5)]
    for test_track_key in test_track_keys:
        test_track_histories.append(track_histories_sum[test_track_key])
        del track_histories_sum[test_track_key]

    len(track_histories_sum.keys())
    s = 0
    for key in track_histories_sum.keys():
        s += len(track_histories_sum[key])
    print(s)

    import numpy as np
    track_histories_sum_numpy = [np.array(v) for k,v in track_histories_sum.items()]
    test_track_histories_numpy = [np.array(v) for v in test_track_histories]

    print("Train: ", len(track_histories_sum_numpy))
    print("Test: ", len(test_track_histories_numpy))




    def generate_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length]
            concat_array = np.concatenate((seq, target.reshape(-1, 2)))
            sequences.append(concat_array)
        return np.array(sequences)

    sequences = []
    for track_history in track_histories_sum_numpy:
        generated_sequences = generate_sequences(track_history, 10)
        if len(generated_sequences) > 0:
            sequences.append(generated_sequences)
    sequences = np.concatenate([seq[:] for seq in sequences], axis=0)


    test_sequences = []
    for track_history in test_track_histories:
        generated_sequences = generate_sequences(track_history, args.seq_len)
        if len(generated_sequences) > 0:
            test_sequences.append(generated_sequences)
    test_sequences = np.concatenate([seq[:] for seq in test_sequences], axis=0)

    test_sequences.shape

    import numpy as np
    import torch
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np

    # Hyperparameters
    learning_rate = args.lr
    epochs = args.epochs
    input_dim = 2  # Assuming only closing price is used
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    sequence_length = 60  # Number of past days used for prediction

    # test_data = torch.tensor(test_sequences)
    # Convert data to tensors
    scaler_train = predictorinterface.MinMaxScalerCustom(-1, 1)
    scaler_test = predictorinterface.MinMaxScalerCustom(-1, 1)


    train_data = torch.tensor(sequences).to(device)
    test_data = torch.tensor(test_sequences).to(device)

    scaler_train.fit(train_data)
    scaler_train.transform(train_data)

    scaler_test.fit(test_data)
    scaler_test.transform(test_data)


    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, -1])
    test_dataset = TensorDataset(test_data[:, :-1], test_data[:, -1])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    #
    model = rnn.LSTMModel(input_dim, hidden_dim, num_layers, 2)
    model = model.to(device)
    # model =
    #
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    test_loss_list = []

    test_loss = rnn.evaluate_model(model, test_loader, criterion, device=device)
    print("Test loss: ", test_loss)
    test_loss_list.append(test_loss)

    for epoch in range(epochs):
        counter = 0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            model = model.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss_list.append(loss.item())
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress (optional)
            if (counter + 1) % 1 == 0:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{counter + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            counter += 1
        test_loss = rnn.evaluate_model(model, test_loader, criterion, device=device)
        print("Test loss: ", test_loss)
        test_loss_list.append(test_loss)

        plt.figure()
        plt.plot(np.arange(len(loss_list)), loss_list)
        plt.plot(np.arange(0, len(loss_list)+1, len(train_loader)), test_loss_list)
        plt.savefig(f"{experiment_name}/plots/loss.png")

        torch.save(model.state_dict(), f'{experiment_name}/checkpoint.pth')

    torch.save(model.state_dict(), f'{experiment_name}/final.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add two numbers.')
    parser.add_argument('--model', type=str, default="lstm", help='First number')
    parser.add_argument('--hidden_dim', type=int, required=True, help='First number')
    parser.add_argument('--num_layers', type=int, help='Second number')
    parser.add_argument('--epochs', type=int, default=100, help='Second number')
    parser.add_argument('--lr', type=float, default=0.001, help='Second number')
    parser.add_argument('--seq_len', type=int, default=10, help='Second number')



    args = parser.parse_args()

    main(args)