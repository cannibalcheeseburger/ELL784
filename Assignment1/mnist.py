import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NeuralNetwork.dense import Dense
from NeuralNetwork.activations import Sigmoid,Softmax
from NeuralNetwork.losses import mse, mse_derive
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


def train(NN, loss, loss_derive, X, Y,x_test,y_test, epoch, alpha, verbose = True):
    error_TS = []
    acc_TS = []
    val_error_TS = []
    val_acc_TS = []
    for e in range(epoch):
        errors= 0
        acc =0
        val_error = 0
        val_acc = 0
        for x, y in zip(X, Y):
            # forward
            out = predict(NN, x)
            if np.argmax(y)==np.argmax(out):
                acc+=1
            # error

            errors+= loss(y, out)

            # backward
            gradient = loss_derive(y, out)
            for layer in reversed(NN):
                gradient = layer.backward(gradient, alpha)

        for x, y in zip(x_test, y_test):
            output = predict(NN, x)
            if np.argmax(y)==np.argmax(out):
                val_acc+=1
            val_error += loss(y, out)
        
        errors/= len(X)
        acc /=len(X)
        val_error /= len(x_test)
        val_acc /=len(x_test)

        if verbose:
            print(f"{e + 1}/{epoch}, error={round(errors,4)}, accuracy={round(acc,4)}, val_error={round(val_error,4)}, val_accuracy={round(val_acc,4)}")
        error_TS.append(errors)
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


metrics = train(NN, mse, mse_derive, X=X_in,Y=Y_in,x_test=X_out,y_test=Y_out, epoch=EPOCHS, alpha=ALPHA)

metrics_names = ['Training Loss','Training Accuracy','Test Loss','Test Accuracy']
for i in range(len(metrics_names)):
    plt.plot(metrics[i])
    plt.xlabel("Epochs")
    plt.ylabel(metrics_names[i])
    plt.savefig("./graph/MNIST_{}.png".format(metrics_names[i]))
    plt.show()