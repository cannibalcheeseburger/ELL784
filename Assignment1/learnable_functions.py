import numpy as np
import pandas as pd
from NeuralNetwork.dense import Dense
from NeuralNetwork.activations import Sigmoid
from NeuralNetwork.losses import mse, mse_derive
from NeuralNetwork.network import train,predict
from tqdm import tqdm

for INPUT in range(1,5):
    def encode(output):
        if output>0.5:
            return 1
        else:
            return 0

    def learnable(X,Y,layer,INPUT):
        X_in = np.reshape(X,X.shape +(1,))
        Y_in = np.reshape(Y,Y.shape +(1,))
        next_in = INPUT
        NN = []
        for l in range(layer):
            NN.append(Dense(next_in,10))
            NN.append(Sigmoid())
            next_in = 10
        NN.append(Dense(next_in, 1))
        NN.append(Sigmoid())

        train(NN, mse, mse_derive, X=X_in,Y=Y_in, epoch=10000, alpha=0.08,verbose=False)
        for x, y in zip(X_in, Y_in):
            output = predict(NN, x)
            if encode(output)!=y:
                return 0
        return 1


    df= pd.read_csv("./boolean/Inputs{}.csv".format(INPUT))
    X,Y = df.iloc[:, :INPUT], df.iloc[:, -1:]
    batch_size = pow(2,INPUT)

    for layer in range(0,4):
        start = 0
        count = 0
        end = batch_size

        batches = pow(2,pow(2,INPUT))
        for batch in tqdm(range(batches),leave=False):
            x_batch = X[start:end]
            y_batch = Y[start:end]
        
            count = count + learnable(X=x_batch,Y=y_batch,layer=layer,INPUT=INPUT)
            #print("Checking Function:{}/{}".format(batch+1,batches))
            start = end
            end  = end+batch_size
        print("Learnable functions in {} Layer(s) for {}-Input(s):{}".format(layer+1,INPUT,count))
        if count==batches:
            break
    
    #ok