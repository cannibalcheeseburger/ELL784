import numpy as np

def mse(true, pred):
    return np.mean(np.power(true - pred, 2))

def mse_derive(true, pred):
    return 2 * (pred - true) / np.size(true)
