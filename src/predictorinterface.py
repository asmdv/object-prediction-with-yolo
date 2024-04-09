import numpy as np
from sklearn.linear_model import LinearRegression
from filterpy.kalman import KalmanFilter

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
        f.P *= 1000.
        f.R = np.array([[5., 0.],
                        [0., 5.]])
        f.Q = np.eye(2) * 0.1

        for i in range(1, past_points.shape[0]):
            z = np.array([[past_points[i][0]], [past_points[i][1]]])
            f.update(z)

        future_points = []
        for i in range(future_t_n):
            f.predict()
            future_points.append(f.x)
        future_points = np.array(future_points).reshape(-1, 1, 2)
        return future_points



