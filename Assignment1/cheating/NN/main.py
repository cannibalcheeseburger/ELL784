import numpy as np
import pandas as pd
from dense import Dense
from activations import Sigmoid
from losses import mse, mse_prime
from network import train
from network import predict

INPUT = 2

def encode(output):
    if output>=0.5:
        return 1
    else:
        return 0

def learnable(X,Y,layers,INPUT):
    X_in = np.reshape(X,X.shape +(1,))
    Y_in = np.reshape(Y,Y.shape +(1,))
    next_in = INPUT
    NN = []
    for layer in range(layers):
        NN.append(Dense(next_in,8))
        NN.append(Sigmoid())
        next_in = 8
    NN.append(Dense(next_in, 1))
    NN.append(Sigmoid())
    train(NN, mse, mse_prime, x_train=X_in,y_train=Y_in, epochs=10000, learning_rate=0.005,verbose=False)
    for x, y in zip(X_in, Y_in):
        output = predict(NN, x)
        if encode(output)!=y:
            return 0
    return 1


df= pd.read_csv("./csv/Inputs{}.csv".format(INPUT+1))
X,Y = df.iloc[:, :INPUT+1], df.iloc[:, -1:]
batch_size = pow(2,INPUT+1)

for layer in range(1,5):
  start = 0
  count = 0
  end = batch_size
  batches = pow(2,pow(2,INPUT+1))
  for batch in range(batches):
    x_batch = X[start:end]
    y_batch = Y[start:end]
    count = count + learnable(X=x_batch,Y=y_batch,layers=layer,INPUT=INPUT+1)
    print("Checking Function:{}/{}".format(batch+1,batches))
    start = end
    end  = end+batch_size
  print("Learnable functions in {} Layer(s):{}".format(layer,count))
  if count==batches:
    break
  
#ok