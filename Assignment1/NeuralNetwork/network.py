import numpy as np 

def predict(NN, input):
    out = input
    for n in NN:
        out = n.forward(out)
    return out



def train(NN, loss, loss_derive, X, Y, epoch, alpha, verbose = True):
    error_TS = []
    acc_TS = []
    for e in range(epoch):
        errors = 0
        acc = 0
        for x, y in zip(X, Y):
            # forward
            out = predict(NN, x)

            if y==out:
                acc+=1
            # error

            errors += loss(y, out)

            # backward
            gradient = loss_derive(y, out)
            for layer in reversed(NN):
                gradient = layer.backward(gradient, alpha)

        errors /= len(X)
        acc /=len(X)
        if verbose:
            print(f"{e + 1}/{epoch}, Train_error={errors}, Train_accuracy={acc}")
        error_TS.append(errors)
        acc_TS.append(acc)
    return error_TS,acc_TS
