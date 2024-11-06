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

def categorical_crossentropy(true,pred):
    pred_clip = np.clip(pred,1e-10,1-1e-10)

    return -1/len(true) * np.sum(np.sum(true * np.log(pred_clip)))

def categorical_crossentropy_derive(true,pred):
    return -true/(pred + 10**-100)


def l0_regularizer_approx(weights, epsilon=1e-5):
    # Sigmoid approximation to count non-zero elements
    return np.sum(1 / (1 + np.exp(-(np.abs(weights) - epsilon) / epsilon)))

# Derivative of the approximate L0 regularizer with respect to weights
def l0_regularizer_approx_derivative(weights, epsilon=1e-5):
    # Sigmoid gradient approximation
    sigmoid_grad = (1 / (1 + np.exp(-(np.abs(weights) - epsilon) / epsilon))) * \
                   (1 - 1 / (1 + np.exp(-(np.abs(weights) - epsilon) / epsilon)))
    # Derivative of |w| is sign(w)
    return sigmoid_grad * np.sign(weights)

# Combined Loss function: BCE + L0 Regularization
def binary_crossentropy_with_l0(y_true, y_pred, weights, lambda_l0):
    bce_loss = binary_crossentropy(y_true, y_pred)
    
    # Compute L0 regularization term (approximation)
    l0_reg = l0_regularizer_approx(weights)
    
    # Total loss
    total_loss = bce_loss + lambda_l0 * l0_reg
    return total_loss

# Combined Gradient: Derivative of BCE + L0
def total_loss_gradient(y_true, y_pred, weights, lambda_l0):
    # Gradient of BCE w.r.t. y_pred
    grad_bce = binary_crossentropy_derive(y_true, y_pred)
    
    # Gradient of L0 approximation w.r.t. weights
    grad_l0 = l0_regularizer_approx_derivative(weights)
    
    # Total gradient (assuming y_pred = f(w, x), so chain rule is needed)
    total_gradient = grad_bce + lambda_l0 * grad_l0
    return total_gradient
