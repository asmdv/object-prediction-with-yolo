import numpy as np
# from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter
import src.rnn
from torch import nn

import torch
from torch.utils.data import TensorDataset, DataLoader

class MinMaxScalerCustom(object):
  """
  MinMaxScaler class for PyTorch tensors.

  This class scales each feature of a tensor to a range between
  new_min (default 0) and new_max (default 1).
  """

  def __init__(self, new_min=0.0, new_max=1.0):
    self.new_min = new_min
    self.new_max = new_max
    self.min_vals = None
    self.max_vals = None

  def fit(self, tensor):
    """
    Computes the minimum and maximum values for each feature.

    Args:
      tensor: A PyTorch tensor of any shape.
    """
    self.min_vals = torch.min(tensor, dim=0, keepdim=True)
    self.max_vals = torch.max(tensor, dim=0, keepdim=True)

  def transform(self, tensor):
    """
    Transforms a tensor by scaling each feature to the specified range.

    Args:
      tensor: A PyTorch tensor of the same shape as the tensor used in `fit`.

    Returns:
      A PyTorch tensor with the same shape as the input tensor, scaled to
      the specified range.
    """
    if self.min_vals is None or self.max_vals is None:
      raise RuntimeError("Please call fit before transform")
    scale = (self.new_max - self.new_min) / (self.max_vals.values - self.min_vals.values)
    return scale * (tensor - self.min_vals.values) + self.new_min

  def inverse_transform(self, tensor):
    """
    Transforms a scaled tensor back to its original range.

    Args:
      tensor: A PyTorch tensor of the same shape as the output of `transform`.

    Returns:
      A PyTorch tensor with the same shape as the input tensor,
      scaled back to its original range.
    """
    if self.min_vals is None or self.max_vals is None:
      raise RuntimeError("Please call fit before transform")
    scale = (self.max_vals.values - self.min_vals.values) / (self.new_max - self.new_min)
    return scale * (tensor - self.new_min) + self.min_vals.values


class PredictorInterface:
    def __int__(self):
        pass

    def predict(self, past_points, future_t_n):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()


class LinearRegressionPredictor(PredictorInterface):
    def __init__(self):
        pass

    def get_info(self):
        d = dict(
            predictor_name=self.__class__.__name__
        )
        return d

    def predict(self, past_points, future_t_n):
        x_points = past_points[:, [0]]
        y_points = past_points[:, [1]]

        t = torch.arange(past_points.shape[0]).reshape(-1, 1)
        t_future = torch.arange(past_points.shape[0], past_points.shape[0] + future_t_n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(t, x_points)
        x_points_future = model.predict(t_future)

        model = LinearRegression()
        model.fit(t, y_points)
        y_points_future = model.predict(t_future)
        future_points = np.hstack((x_points_future, y_points_future)).reshape((-1, 1, 2))
        future_points = torch.from_numpy(future_points).to(torch.float32)
        return future_points



class KalmanFilterPredictor(PredictorInterface):
    def __init__(self):
        pass

    def predict(self, past_points, future_t_n):
        f = KalmanFilter(dim_x=2, dim_z=2)
        f.x = np.array([[past_points[0][0]],
                        [past_points[0][1]]])
        f.F = np.array([[1., 0.],
                        [0., 1.]])
        f.H = np.array([[1., 0.],
                        [0., 1.]])
        f.P = np.cov(past_points.reshape(2, -1))
        f.R = np.array([[5., 0.],
                        [0., 5.]])
        f.Q = np.eye(2) * 0.1

        for i in range(1, past_points.shape[0]):
            z = np.array([[past_points[i][0]], [past_points[i][1]]])
            print("Past:", z)
            f.update(z)

        future_points = []
        for i in range(future_t_n):
            f.predict()
            print("Future:", f.x)
            future_points.append(f.x)
        return
        future_points = np.array(future_points).reshape(-1, 1, 2)
        return future_points


class LSTMPredictor(PredictorInterface):
    def __init__(self, hidden_dim_size=32, num_layers=1, lr=0.001, epochs=100):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        pass

    def get_info(self):
        s = f"{self.__class__.__name__},n_layers:{self.num_layers},hidden_size:{self.hidden_dim_size},lr:{self.lr},epochs:{self.epochs}"
        d = dict(
            predictor_name=self.__class__.__name__,
            n_layers=self.num_layers,
            hidden_size=self.hidden_dim_size,
            lr=self.lr,
            epochs=self.epochs
        )
        return d

    def predict(self, past_points, future_t_n):
        scaler = MinMaxScalerCustom(-1, 1)
        scaler.fit(past_points)
        scaler.transform(past_points)
        past_points = scaler.transform(past_points)
        past_points = src.rnn.create_windowed_arrayV2(past_points, future_t_n)
        train_data = past_points
        train_dataset = TensorDataset(train_data[:, :-1], train_data[:, -1])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        model = src.rnn.LSTMModel(2, self.hidden_dim_size, self.num_layers, 2)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                outputs = model(data)
                loss = criterion(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        predicted = src.rnn.predict(model, train_data[-future_t_n:])
        predicted = scaler.inverse_transform(predicted.detach().numpy())
        return predicted.unsqueeze(1)