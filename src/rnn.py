import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset



def create_windowed_array(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return np.array(windows)

def create_windowed_arrayV2(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size])
    return torch.stack(windows)


def get_google_stock_data(start_date, end_date, split_ratio=0.8):
  google_data = yf.download("GOOG", start=start_date, end=end_date)

  data = google_data['Close']
  # Scale or normalize the data (replace with your preferred method)
  data = create_windowed_array(data, 10)

  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaled_data = scaler.fit_transform(data)

  # Split data into training and testing sets
  num_datapoints = len(scaled_data)
  split_index = int(num_datapoints * split_ratio)

  train_data = scaled_data[:split_index]
  test_data = scaled_data[split_index:]

  return train_data, test_data, scaler, num_datapoints, split_index  # Include scaler for inverse transformation later

# # Example usage
# class LinearRegression(nn.Module):
#     def __init__(self):
#         super(LinearRegression, self).__init__()
#         self.linear = nn.Linear(2, 2)  # One input feature, one output
#
#     def forward(self, x):
#         return self.linear(x)


# Define LSTM model
class LSTMModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
    super(LSTMModel, self).__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
    out, (hn, cn) = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

class LSTMModelV2(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim, n_predictions, device):
    super(LSTMModelV2, self).__init__()
    self.device = device
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.output_dim = output_dim
    self.n_predictions = n_predictions
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
      batch_size = x.size(0)
      h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
      c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)

      outputs = []
      input_sequence = x

      for _ in range(self.n_predictions):
          out, (hn, cn) = self.lstm(input_sequence, (h0, c0))
          h0, c0 = hn, cn
          out = self.fc(out[:, -1, :])
          outputs.append(out.unsqueeze(1))
          input_sequence = torch.cat([x, out.unsqueeze(1)], dim=1)

      outputs = torch.cat(outputs, dim=1)
      return outputs

  # def forward(self, x):
  #   batch_size = x.size(0)
  #   h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
  #   c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
  #   out, (hn, cn) = self.lstm(x, (h0, c0))
  #
  #   last_hidden_state = out[:, -1, :]
  #   print("Last hidden state shape", last_hidden_state.unsqueeze(0).shape)
  #   print("hn shape", hn.shape)
  #   print("cn shape", cn.shape)
  #   predictions = torch.zeros(batch_size, self.n_predictions, self.output_dim).to(device)
  #   # Perform iterative prediction for n steps
  #   for i in range(self.n_predictions):
  #       print("Predictions shape", predictions[:, i:i+1,:].shape)
  #       # Pass the last hidden state as input for future prediction
  #       pred_i, (hn, cn) = self.lstm(predictions[:, i:i+1, :], (last_hidden_state.unsqueeze(0), cn))  # Unsqueeze for batch dimension
  #       predictions[:, i, :] = self.fc(pred_i[:, -1, :])
  #       last_hidden_state = hn[:, -1, :]
  #
  #   return predictions


# Predict on new data (optional)
def predict(model, data):
  if len(list(data.shape)) == 2:
    data = data.unsqueeze(-1)
  prediction = model(data)
  return prediction

def evaluate_modelV2(model, data_loader, criterion, device):
  # model = model.to(device)
  model.eval()  # Set the model to evaluation mode
  total_loss = 0.0
  total_loss_x = 0.0
  total_loss_y = 0.0
  data_loader_iter = iter(data_loader)
  _, target = next(data_loader_iter)
  losses = torch.zeros(3, target.size(1), device=device)
  with torch.no_grad():  # Disable gradient calculation for efficiency during testing
    for data, target in data_loader:
      # data = data.unsqueeze(-1)
      data, target = data.to(device), target.to(device)
      outputs = model(data)
      for i in range(outputs.size(1)):
          losses[0, i] += criterion(outputs[:, i, :], target[:, i, :])
          losses[1, i] += criterion(outputs[:, i, 0], target[:, i, 0])
          losses[2, i] += criterion(outputs[:, i, 1], target[:, i, 1])

      # total_loss += loss.item()
      # total_loss_x += loss_x.item()
      # total_loss_y += loss_y.item()
  losses = losses / len(data_loader)
  return losses[0, :].mean().item(), (losses[1, :].mean().item(), losses[2, :].mean().item()), losses # / len(data_loader), (total_loss_x / len(data_loader), total_loss_y / len(data_loader))


def evaluate_model(model, data_loader, criterion, device=torch.device('cpu')):
  model = model.to(device)
  model.eval()  # Set the model to evaluation mode
  total_loss = 0.0
  total_loss_x = 0.0
  total_loss_y = 0.0
  with torch.no_grad():  # Disable gradient calculation for efficiency during testing
    for data, target in data_loader:
      # data = data.unsqueeze(-1)
      data, target = data.to(device), target.to(device)
      outputs = model(data)
      loss = criterion(outputs, target)
      loss_x = criterion(outputs[:, 0], target[:, 0])
      loss_y = criterion(outputs[:, 1], target[:, 1])

      total_loss += loss.item()
      total_loss_x += loss_x.item()
      total_loss_y += loss_y.item()
  return total_loss / len(data_loader), (total_loss_x / len(data_loader), total_loss_y / len(data_loader))


def main():
    import numpy as np
    train_data, test_data, scaler, num_points, split_index = get_google_stock_data("2020-01-01", "2024-04-25")
    plt.plot(np.arange(num_points)[:split_index], train_data[:, 0])
    plt.plot(np.arange(num_points)[split_index:], test_data[:, 0])

    # Hyperparameters
    learning_rate = 0.001
    epochs = 100
    input_dim = 1  # Assuming only closing price is used
    hidden_dim = 32
    num_layers = 1
    sequence_length = 60  # Number of past days used for prediction

    # Convert data to tensors
    train_data = torch.tensor(train_data).float()
    test_data = torch.tensor(test_data).float()

    # print(train_data)
    # Create training and testing datasets
    train_dataset = TensorDataset(train_data[:, :-1], train_data[:, -1])
    test_dataset = TensorDataset(test_data[:, :-1], test_data[:, -1])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(input_dim, hidden_dim, num_layers, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(-1)
            outputs = model(data)
            loss = criterion(outputs, target.reshape(-1, 1))
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress (optional)
            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    # Evaluate the model on test data
    # ...
    test_loss, _ = evaluate_model(model, test_loader, criterion)

    output = predict(model, test_data[:, :-1])
    output = output.detach().numpy()
    plt.plot(np.arange(num_points)[split_index:], output)
    plt.show()
    print("Test loss", test_loss)


# Define the dataset class for 2D points
class TimeSeriesDataset2D(Dataset):
    def __init__(self, data, input_length, output_length):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.input_length]
        output_seq = self.data[idx + self.input_length:idx + self.input_length + self.output_length]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)


# Define the linear regression model for 2D points
class LinearRegressionModel2D(nn.Module):
    def __init__(self, input_length, output_length):
        super(LinearRegressionModel2D, self).__init__()
        self.linear = nn.Linear(input_length * 2, output_length * 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        output = self.linear(x)
        return output.view(batch_size, -1, 2)  # Reshape the output




if __name__ == "__main__":
    main()