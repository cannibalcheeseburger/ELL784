import numpy as np 
def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output



def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True,categorical=False):
    error_TS = []
    acc_TS = []
    for e in range(epochs):
        error = 0
        acc =0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            if categorical:
                if np.argmax(y)==np.argmax(output):
                    acc+=1
            else:
                if y==output:
                    acc+=1
            # error

            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        acc /=len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}, accuracy={acc}")
        error_TS.append(error)
        acc_TS.append(acc)
    return error_TS,acc_TS
