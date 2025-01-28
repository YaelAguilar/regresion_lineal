import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * X_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.slope * X + self.intercept