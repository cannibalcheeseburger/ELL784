import numpy as np

def mse(true, pred):
    return np.mean(np.power(true - pred, 2))

def mse_derive(true, pred):
    return 2 * (pred - true) / np.size(true)

def binary_crossentropy(true,pred):
    pred_clip = np.clip(pred,1e-10,1-1e-10)
    return -np.mean(true*np.log(pred_clip)+(1-true)*np.log(1-pred_clip))

def binary_crossentropy_derive(true,pred):
    pred_clip = np.clip(pred,1e-10,1-1e-10)
    return (pred_clip-true)/(pred_clip*(1-pred_clip))