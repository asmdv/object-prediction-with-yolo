import numpy as np
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter
import src.rnn
from torch import nn
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

class PredictorInterface:
    def __int__(self):
        pass

    def predict(self, past_points, future_t_n):
        raise NotImplementedError()

class LinearRegressionPredictor(PredictorInterface):
    def __init__(self):
        pass
    def predict(self, past_points, future_t_n):
        x_points = past_points[:, [0]]
        y_points = past_points[:, [1]]

        t = np.arange(past_points.shape[0]).reshape(-1, 1)
        t_future = np.arange(past_points.shape[0], past_points.shape[0] + future_t_n).reshape(-1, 1)

        model = LinearRegression()
        model.fit(t, x_points)
        x_points_future = model.predict(t_future)

        model = LinearRegression()
        model.fit(t, y_points)
        y_points_future = model.predict(t_future)
        future_points = np.hstack((x_points_future, y_points_future)).reshape((-1, 1, 2))
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
    def __init__(self):
        pass
    def predict(self, past_points, window_size):
        print(past_points)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        obj = scaler.fit(past_points)
        past_points = scaler.fit_transform(past_points)

        past_points = src.rnn.create_windowed_array(past_points, window_size)
        print(past_points)
        train_data = torch.tensor(past_points).float()
        print(train_data.shape)
        train_dataset = TensorDataset(train_data[:, :-1], train_data[:, -1])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        learning_rate = 0.001
        epochs = 100

        model = src.rnn.LSTMModel(2, 32, 1, 2)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for i, (data, target) in enumerate(train_loader):
                outputs = model(data)
                loss = criterion(outputs, target)
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print training progress (optional)
                if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        predicted = src.rnn.predict(model, train_data[-window_size:])
        predicted = predicted.detach().numpy()
        predicted = obj.inverse_transform(predicted)
        return predicted.reshape(-1, 1, 2)