import pickle
import argparse
import os
import src.rnn as rnn
import datetime
import sys
import src.predictorinterface as predictorinterface
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

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

def get_pytorch_device():
  """
  Identifies the available PyTorch device (CUDA, MPS, or CPU).

  Returns:
      str: The name of the available device ('cuda', 'mps', or 'cpu').
  """

  # Check for CUDA availability
  if torch.cuda.is_available():
    return 'cuda'

  # Check for MPS availability (if applicable)
  if torch.backends.mps.is_available():
    return 'mps'

  # Default to CPU
  return 'cpu'


def main(args):
    experiment_name = create_experiment_path(args)

    f = open(f'{experiment_name}/log.txt', 'a')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    print(args)
    device = get_pytorch_device()
    print(f"Using device: {device}")
    # track_history_paths = [f"pickle/track_history_amsterdam_full_000{i}.mp4.pkl" for i in range(7)]
    # track_histories = []
    # track_histories_sum = {}
    #
    #
    # for counter, track_history_path in enumerate(track_history_paths):
    #     print(f"Loading {track_history_path}")
    #     with open(track_history_path, 'rb') as file:
    #         dict_with_changed_keys = change_keys(pickle.load(file), lambda x: add_file_variable_name(x, counter))
    #         track_histories.append(dict_with_changed_keys)
    #
    # for track_history in track_histories:
    #     Merge(track_histories_sum, track_history)
    #
    #
    # val_track_histories = []
    # val_track_keys = [(2470, 'truck', 6),
    #                    (2410, 'person', 6),
    #                    (555, 'motorcycle', 6),
    #                    (492, 'car', 6),
    #                    (85, 'person', 6),
    #                    (3690, 'car', 5),
    #                    (3539, 'person', 5)]
    # for val_track_key in val_track_keys:
    #     val_track_histories.append(track_histories_sum[val_track_key])
    #     del track_histories_sum[val_track_key]
    #
    # test_track_histories = []
    # test_track_keys = [
    #     (1642, 'bicycle', 2),
    #     (1049, 'person', 4),
    #     (3059, 'person', 0),
    #     (449, 'bicycle', 5),
    #     (1241, 'motorcycle', 0),
    #     (645, 'umbrella', 0),
    #     (1101, 'person', 3),
    #     (430, 'car', 3),
    #     (685, 'bench', 1),
    #     (1194, 'chair', 6),
    #     (3225, 'person', 5)
    # ]
    #
    # for test_track_key in test_track_keys:
    #     test_track_histories.append(track_histories_sum[test_track_key])
    #     del track_histories_sum[test_track_key]

    with open(args.data_path, "rb") as f:
        track_histories_sum = pickle.load(f)

    s = 0
    count_keys = 0
    for key in track_histories_sum.keys():
        count_keys += 1
        s += len(track_histories_sum[key])
    print("Number of objects:", count_keys)
    print("Number of training points:", s)

    track_histories_sum_numpy = [np.array(v) for k,v in track_histories_sum.items()]

    def generate_sequences(data, sequence_length, future_length):
        sequences = []
        for i in range(len(data) - sequence_length - future_length):
            seq = data[i:i+sequence_length]
            target = data[i+sequence_length:i+sequence_length+future_length]
            concat_array = np.concatenate((seq, target))
            sequences.append(concat_array)
        return np.array(sequences)

    sequences = []
    for track_history in track_histories_sum_numpy:
        generated_sequences = generate_sequences(track_history, args.seq_len, args.seq_len)
        if len(generated_sequences) > 0:
            sequences.append(generated_sequences)
    sequences = np.concatenate([seq[:] for seq in sequences], axis=0)

    print("Sequences:", sequences.shape)

    def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Splits a NumPy array into train, validation, and test sets.

        Args:
            data (np.ndarray): The NumPy array to split.
            train_ratio (float, optional): Proportion of data for training set. Defaults to 0.8.
            val_ratio (float, optional): Proportion of data for validation set. Defaults to 0.1.
            test_ratio (float, optional): Proportion of data for test set. Defaults to 0.1.

        Returns:
            tuple: A tuple containing the train, validation, and test sets as NumPy arrays.
        """

        # Ensure ratios sum to 1
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError(
                "Ratios must sum to 1. Provided ratios: train={}, val={}, test={}".format(train_ratio, val_ratio,
                                                                                          test_ratio))

        data_size = len(data)
        train_size = int(data_size * train_ratio)
        val_size = int(data_size * val_ratio)
        test_size = data_size - train_size - val_size

        # Randomly shuffle the data for better generalization
        shuffled_indices = np.random.permutation(data_size)
        shuffled_data = data[shuffled_indices]

        train_set = shuffled_data[:train_size]
        val_set = shuffled_data[train_size:train_size + val_size]
        test_set = shuffled_data[train_size + val_size:]

        return train_set, val_set, test_set

    train_sequences, val_sequences, test_sequences = split_data(sequences)
    print("Number of train sequences:", len(train_sequences))
    print("Number of validation sequences:", len(val_sequences))
    print("Number of test sequences:", len(test_sequences))

    # val_sequences = []
    # for track_history in val_track_histories_numpy:
    #     generated_sequences = generate_sequences(track_history, args.seq_len, args.seq_len)
    #     if len(generated_sequences) > 0:
    #         val_sequences.append(generated_sequences)
    # val_sequences = np.concatenate([seq[:] for seq in val_sequences], axis=0)
    #
    # test_sequences = []
    # for track_history in test_track_histories_numpy:
    #     generated_sequences = generate_sequences(track_history, args.seq_len, args.seq_len)
    #     if len(generated_sequences) > 0:
    #         test_sequences.append(generated_sequences)
    # test_sequences = np.concatenate([seq[:] for seq in test_sequences], axis=0)


    # Hyperparameters
    learning_rate = args.lr
    epochs = args.epochs
    input_dim = 2  # Assuming only closing price is used
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    sequence_length = 60  # Number of past days used for prediction

    # Convert data to tensors
    scaler_train = predictorinterface.MinMaxScalerCustom(-1, 1)
    scaler_val = predictorinterface.MinMaxScalerCustom(-1, 1)
    scaler_test = predictorinterface.MinMaxScalerCustom(-1, 1)



    train_data = torch.tensor(train_sequences).to(device)
    val_data = torch.tensor(val_sequences).to(device)
    test_data = torch.tensor(test_sequences).to(device)

    scaler_train.fit(train_data)
    scaler_train.transform(train_data)

    scaler_val.fit(val_data)
    scaler_val.transform(val_data)

    scaler_test.fit(test_data)
    scaler_test.transform(test_data)


    train_dataset = TensorDataset(train_data[:, :args.seq_len], train_data[:, args.seq_len:args.seq_len + args.seq_len])
    val_dataset = TensorDataset(val_data[:, :args.seq_len], val_data[:, args.seq_len: args.seq_len + args.seq_len])
    test_dataset = TensorDataset(test_data[:, :args.seq_len], test_data[:, args.seq_len: args.seq_len + args.seq_len])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    #
    model = rnn.LSTMModelV2(input_dim, hidden_dim, num_layers, 2, n_predictions=args.seq_len, device=device)
    model = model.to(device)
    # model =
    #
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    val_loss_list = []

    extended_val_loss_list = []

    val_loss, (val_loss_x, val_loss_y), loss_tensor = rnn.evaluate_modelV2(model, val_loader, criterion, device=device)
    print("Validation loss: ", val_loss)
    print("Validation loss x: ", val_loss_x)
    print("Validation loss y: ", val_loss_y)

    val_loss_list.append(val_loss)
    extended_val_loss_list.append(loss_tensor.unsqueeze(0))

    for epoch in range(epochs):
        counter = 0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # model = model.to(device)
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

        val_loss, (val_loss_x, val_loss_y), loss_tensor = rnn.evaluate_modelV2(model, val_loader, criterion,
                                                                               device=device)
        print("Validation loss: ", val_loss)
        print("Validation loss x: ", val_loss_x)
        print("Validation loss y: ", val_loss_y)

        val_loss_list.append(val_loss)
        extended_val_loss_list.append(loss_tensor.unsqueeze(0))
        plt.figure()
        plt.plot(np.arange(len(loss_list)), loss_list)
        plt.plot(np.arange(0, len(loss_list)+1, len(train_loader)), val_loss_list)
        plt.savefig(f"{experiment_name}/plots/loss.png")
        torch.save(model.state_dict(), f'{experiment_name}/checkpoint.pth')
        concat_loss_list = torch.cat(extended_val_loss_list, dim=0)
        torch.save({"loss_list": loss_list, "val_loss_list": val_loss_list, "len_train_loader": len(train_loader), "extended_val_loss_list": concat_loss_list}, f'{experiment_name}/loss.dat')

        concat_loss_list = concat_loss_list.cpu()
        plt.figure()
        for i in range(concat_loss_list.size(2)):
            print(i)
            plt.plot(np.arange(1, concat_loss_list.size(0) + 1), concat_loss_list[:, 0, i], label=f"t={i + 1}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
        plt.savefig(f"{experiment_name}/plots/val_loss_per_t.png")


    torch.save(model.state_dict(), f'{experiment_name}/final.pth')
    test_loss, (test_loss_x, test_loss_y), loss_tensor = rnn.evaluate_modelV2(model, test_loader, criterion, device=device)
    torch.save({"loss_list": loss_list, "val_loss_list": val_loss_list, "len_train_loader": len(train_loader),
                "extended_val_loss_list": torch.cat(extended_val_loss_list, dim=0), "extended_test_loss": loss_tensor}, f'{experiment_name}/loss_final.dat')
    print("Test loss:", test_loss)
    print("Test loss x:", test_loss_x)
    print("Test loss y:", test_loss_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add two numbers.')
    parser.add_argument('--model', type=str, default="lstm", help='First number')
    parser.add_argument('--hidden_dim', type=int, required=True, help='First number')
    parser.add_argument('--num_layers', type=int, help='Second number')
    parser.add_argument('--epochs', type=int, default=100, help='Second number')
    parser.add_argument('--lr', type=float, default=0.001, help='Second number')
    parser.add_argument('--seq_len', type=int, default=10, help='Second number')
    parser.add_argument('--data_path', type=str, required=True)


    args = parser.parse_args()

    main(args)