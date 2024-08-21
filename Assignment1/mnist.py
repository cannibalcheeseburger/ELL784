import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NeuralNetwork.dense import Dense
from NeuralNetwork.activations import Sigmoid,Softmax
from NeuralNetwork.losses import mse, mse_prime
from NeuralNetwork.network import predict
from tqdm import tqdm

EPOCHS = 150
ALPHA = 0.03

NN = [Dense(784,200),
      Sigmoid(),
      Dense(200,40),
      Sigmoid(),
      Dense(40,10),
      Softmax()
      ]


def train(network, loss, loss_prime, x_train, y_train,x_test,y_test, epochs = 1000, learning_rate = 0.01, verbose = True):
    error_TS = []
    acc_TS = []
    val_error_TS = []
    val_acc_TS = []
    for e in range(epochs):
        error = 0
        acc =0
        val_error = 0
        val_acc = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)
            if np.argmax(y)==np.argmax(output):
                acc+=1
            # error

            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        for x, y in zip(x_test, y_test):
            output = predict(network, x)
            if np.argmax(y)==np.argmax(output):
                val_acc+=1
            val_error += loss(y, output)
        
        error /= len(x_train)
        acc /=len(x_train)
        val_error /= len(x_test)
        val_acc /=len(x_test)

        if verbose:
            print(f"{e + 1}/{epochs}, error={round(error,4)}, accuracy={round(acc,4)}, val_error={round(val_error,4)}, val_accuracy={round(val_acc,4)}")
        error_TS.append(error)
        acc_TS.append(acc)
        val_error_TS.append(val_error)
        val_acc_TS.append(val_acc)
    return [error_TS,acc_TS,val_error_TS,val_acc_TS]

df_train = pd.read_csv("./dataset/MNIST.csv")

Y_test,X_test =  df_train.iloc[:2000, :1], df_train.iloc[:2000, 1:]
Y_train,X_train = df_train.iloc[2000:, :1], df_train.iloc[2000:, 1:]
X_in = np.reshape(X_train,X_train.shape +(1,))
Y_in = np.squeeze(np.eye(10)[Y_train])
Y_in = np.reshape(Y_in,Y_in.shape +(1,))
X_out = np.reshape(X_test,X_test.shape +(1,))
Y_out = np.squeeze(np.eye(10)[Y_test])
Y_out = np.reshape(Y_out,Y_out.shape +(1,))


metrics = train(NN, mse, mse_prime, x_train=X_in,y_train=Y_in,x_test=X_out,y_test=Y_out, epochs=EPOCHS, learning_rate=ALPHA)

metrics_names = ['Training Loss','Training Accuracy','Test Loss','Test Accuracy']
for i in range(len(metrics_names)):
    plt.plot(metrics[i])
    plt.xlabel("Epochs")
    plt.ylabel(metrics_names[i])
    plt.savefig("./graph/MNIST_{}.png".format(metrics_names[i]))
    plt.show()