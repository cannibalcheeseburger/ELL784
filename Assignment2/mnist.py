import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from NeuralNetwork.dense import Dense
from NeuralNetwork.activations import Sigmoid,Softmax
from NeuralNetwork.losses import mse, mse_derive
from NeuralNetwork.network import predict
from tqdm import tqdm
import pickle

EPOCHS = 1
ALPHA = [0.002,0.008,0.03,0.08,0.2]

def k_fold(X,Y,k,set_fold):
    batch = int(len(X)/k)
    start = int(set_fold*batch)
    end = int((set_fold+1)*batch)
    x_val = X[start:end]
    y_val = Y[start:end]
    x_train = np.concatenate((X[:start],X[end:]))
    y_train = np.concatenate((Y[:start],Y[end:]))
    return x_train,y_train,x_val,y_val

def train(NN, loss, loss_derive, X, Y,epoch, alpha,k = 10, verbose = True):
    error_TS = []
    acc_TS = []
    val_error_TS = []
    val_acc_TS = []
    for e in range(epoch):
        errors= 0
        acc =0
        val_error = 0
        val_acc = 0
        set_fold = e%k
        x_train,y_train,x_val,y_val = k_fold(X,Y,k,set_fold)
        for x, y in zip(x_train, y_train):
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

        for x, y in zip(x_val, y_val):
            output = predict(NN, x)
            if np.argmax(y)==np.argmax(output):
                val_acc+=1
            val_error += loss(y, output)
        
        errors/= len(x_train)
        acc /=len(x_train)
        val_error /= len(x_val)
        val_acc /=len(x_val)

        if verbose:
            print(f"{e + 1}/{epoch}, error={round(errors,4)}, accuracy={round(acc,4)}, val_error={round(val_error,4)}, val_accuracy={round(val_acc,4)}")
        error_TS.append(errors)
        acc_TS.append(acc)
        val_error_TS.append(val_error)
        val_acc_TS.append(val_acc)
    return [error_TS,acc_TS,val_error_TS,val_acc_TS]

def testModel(NN, loss,x_test,y_test):
    test_error = 0
    test_acc = 0
    for x, y in zip(x_test, y_test):
        output = predict(NN, x)
        if np.argmax(y)==np.argmax(output):
            test_acc+=1
        test_error += loss(y, output)
    return test_error/len(x_test),test_acc/len(x_test)


df_train = pd.read_csv("./dataset/MNIST.csv")

Y_test,X_test =  df_train.iloc[:2000, :1], df_train.iloc[:2000, 1:]
Y_train,X_train = df_train.iloc[2000:, :1], df_train.iloc[2000:, 1:]
X_in = np.reshape(X_train,X_train.shape +(1,))
Y_in = np.squeeze(np.eye(10)[Y_train])
Y_in = np.reshape(Y_in,Y_in.shape +(1,))
X_test = np.reshape(X_test,X_test.shape +(1,))
Y_test = np.squeeze(np.eye(10)[Y_test])
Y_test = np.reshape(Y_test,Y_test.shape +(1,))

export = {}

for alpha in ALPHA:
    NN = [Dense(784,200),
        Sigmoid(),
        Dense(200,40),
        Sigmoid(),
        Dense(40,10),
        Softmax()
        ]
    metrics = train(NN, mse, mse_derive, X=X_in,Y=Y_in, epoch=EPOCHS, alpha=alpha,k=10)
    test_error,test_acc = testModel(NN,mse,X_test,Y_test)
    metrics.append(test_error)
    metrics.append(test_acc)
    export[str(alpha)] = metrics
    metrics_names = ['Training Loss','Training Accuracy','Validation Loss','Validation Accuracy',"Test Loss","Test Accuracy"]
    for i in range(len(metrics_names)):
        plt.plot(metrics[i])
        plt.xlabel("Epochs")
        plt.ylabel(metrics_names[i])
        plt.savefig("./graph/MNIST_{}_{}.png".format(metrics_names[i],alpha))
        plt.clf()
    
with open('Export','wb') as f:
    pickle.dump(export,f)