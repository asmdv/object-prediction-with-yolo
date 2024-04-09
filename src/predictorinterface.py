from sklearn.linear_model import LinearRegression

class PredictorInterface:
    def __int__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

class LinearRegressionPredictor(PredictorInterface):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        return self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

