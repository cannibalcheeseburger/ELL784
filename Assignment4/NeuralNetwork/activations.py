import numpy as np

class Sigmoid():
    def __init__(self):
        def sigmoid(z):
            z = np.clip(z, -100, 100)

            sig = 1 + np.exp(-z)
            return 1 / sig

        def sigmoid_prime(z):
            sig = sigmoid(z)
            return sig * (1 - sig)
        
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime
        
    def forward(self, inp):
        self.input = inp
        return self.activation(self.input)

    def backward(self, output_grad, alpha):
        return np.multiply(output_grad, self.activation_prime(self.input))


class ReLU():
    def __init__(self):
        def reLU(x):
            return np.maximum(0,x)

        def reLU_prime(x):
            return (x>0)*1
        
        self.activation = reLU
        self.activation_prime = reLU_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, alpha):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Softmax():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inp):
        tmp = inp - np.max(inp)
        tmp = np.exp(tmp)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_grad, alpha):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_grad)
