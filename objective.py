import numpy as np

def mean_squared_error(beta, X, Y):
    predictions = beta[0] + np.dot(X, beta[1:])
    mse = np.mean((Y - predictions) ** 2)
    return mse
