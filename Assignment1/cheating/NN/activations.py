import numpy as np

class Sigmoid():
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        self.activation = sigmoid
        self.activation_prime = sigmoid_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

        #super().__init__(sigmoid, sigmoid_prime)

class reLU():
    def __init__(self):
        def reLU(x):
            return np.maximum(0,x)

        def reLU_prime(x):
            return x>0
        
        self.activation = reLU
        self.activation_prime = reLU_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Softmax():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
