import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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

# Example usage


# Define LSTM model
class LSTMModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
    super(LSTMModel, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    batch_size = x.size(0)
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
    out, (hn, cn) = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

# Predict on new data (optional)
def predict(model, data):
  if len(list(data.shape)) == 2:
    data = data.unsqueeze(-1)
  prediction = model(data)
  return prediction

def evaluate_model(model, data_loader, criterion):
  model.eval()  # Set the model to evaluation mode
  total_loss = 0.0
  with torch.no_grad():  # Disable gradient calculation for efficiency during testing
    for data, target in data_loader:
      # data = data.unsqueeze(-1)
      outputs = model(data)
      loss = criterion(outputs, target)
      total_loss += loss.item()
  return total_loss / len(data_loader)


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
    test_loss = evaluate_model(model, test_loader, criterion)

    output = predict(model, test_data[:, :-1])
    output = output.detach().numpy()
    plt.plot(np.arange(num_points)[split_index:], output)
    plt.show()
    print("Test loss", test_loss)


if __name__ == "__main__":
    main()