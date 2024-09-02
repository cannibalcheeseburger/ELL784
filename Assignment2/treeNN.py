from NeuralNetwork.dense import Dense
from NeuralNetwork.losses import mse,mse_derive
from NeuralNetwork.activations import Sigmoid,Softmax
from NeuralNetwork.network import predict
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image


EPOCH = 30
ALPHA = 0.08

def encode(z):
    if z>=0.5:
        return 1
    return 0

def load_mnist_images_to_df(filename):
    with open(filename, 'rb') as f:
        _ = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        image_data = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols)
        df = pd.DataFrame(images)
    return df

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        label_data = f.read(num_labels)
        labels = np.frombuffer(label_data, dtype=np.uint8)        
    return labels


def train(NN, loss, loss_derive, X, Y, X_val,Y_val, epoch=EPOCH, alpha=ALPHA, verbose = True):
    error_TS = []
    acc_TS = []
    val_acc_TS = []
    val_error_TS = []
    for e in range(epoch):
        errors = 0
        acc = 0
        val_acc = 0
        val_error = 0
        for x, y in zip(X, Y):
            out = predict(NN, x)
            if y==encode(out):
                acc+=1
            errors += loss(y, out)
            gradient = loss_derive(y, out)
            for layer in reversed(NN):
                gradient = layer.backward(gradient, alpha)

        for x, y in zip(X_val, Y_val):
            output = predict(NN, x)
            if y==encode(output):
                val_acc+=1
            val_error += loss(y, output)

        errors /= len(X)
        acc /=len(X)
        val_error /= len(X_val)
        val_acc /=len(X_val)
        if verbose:
            print(f"{e + 1}/{epoch}, error={round(errors,4)}, accuracy={round(acc,4)}, val_error={round(val_error,4)}, val_accuracy={round(val_acc,4)}")
        error_TS.append(errors)
        acc_TS.append(acc)
        val_error_TS.append(val_error)
        val_acc_TS.append(val_acc)
    return [error_TS,acc_TS,val_error_TS,val_acc_TS]

class TreeNN():
    def __init__(self,depth=1,max_depth=3,parent_node = None):
        self.parent_node = parent_node
        self.NN = [
            Dense(784,1),
            Sigmoid()
        ]
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
    
    def get_augemented(self,X,Y):
        Xa_train = []
        Xb_train = []
        for x, y in zip(X, Y):
            output = encode(predict(self.NN, x))
            if y!=output:
                if output==0 and y==1:
                    Xa_train.append(1)
                    Xb_train.append(0)
                if output==1 and y==0:
                    Xa_train.append(0)
                    Xb_train.append(1)

            else:
                if output==0 and y==0:
                    Xa_train.append(0)
                    Xb_train.append(1)

                if output==1 and y==1:
                    Xa_train.append(1)
                    Xb_train.append(0)
        return Xa_train,Xb_train
    
    def plot(self,metrics):
        metrics_names = ['Training Loss','Training Accuracy','Validation Loss','Validation Accuracy']
        for i in range(len(metrics_names)):
            plt.plot(metrics[i])
            plt.xlabel("Epochs")
            plt.ylabel(metrics_names[i])
            plt.savefig("./graph/MNIST_{}_{}.png".format(metrics_names[i],self.depth))
            plt.clf()


    
    def train_tree(self,X,Y,X_val,Y_val):
        print("Layers :",self.depth)
        X_train = np.reshape(X,X.shape +(1,))
        Y_train = np.reshape(Y,Y.shape +(1,))
        X_val_in =np.reshape(X_val,X_val.shape +(1,))
        Y_val_in =np.reshape(Y_val,Y_val.shape +(1,))
        metrics = train(self.NN,mse,mse_derive,X_train,Y_train,X_val_in,Y_val_in)
        self.plot(metrics)
        if self.max_depth>self.depth+1:
            ## Adding Layers
            Na = TreeNN(depth=self.depth+1,max_depth=self.max_depth,parent_node=self)
            Nb = TreeNN(depth=self.depth+1,max_depth=self.max_depth,parent_node=self)
            self.children.append(Na)
            self.children.append(Nb)
            #Retrain with extra feature
            Xa_train,Xb_train = self.get_augemented(X_train,Y_train)
            X_retrain = np.column_stack((X, Xa_train))
            X_retrain = np.column_stack((X_retrain, Xb_train))
            X_retrain = np.reshape(X_retrain,X_retrain.shape+(1,))
            Xa,Xb = self.get_augemented(X_val_in,Y_val_in)
            X_reval = np.column_stack((X_val, Xa))
            X_reval = np.column_stack((X_reval, Xb))
            X_reval = np.reshape(X_reval,X_reval.shape+(1,))
            self.NN = [Dense(786,1),Sigmoid()]
            metrics = train(self.NN,mse,mse_derive,X_retrain,Y_train,X_reval,Y_val_in)
            self.plot(metrics)
            #Na
            self.children[0].train_tree(X,np.asanyarray(Xa_train),X_val,np.asanyarray(Xa))
            #Nb
            self.children[1].train_tree(X,np.asanyarray(Xb_train),X_val,np.asanyarray(Xb))

    def predict(self,X):
        if self.depth>self.max_depth:
            return  None  
        predictions = []

        if self.children == []:
            X_train = np.reshape(X,X.shape +(1,))
        else:
            Xa = self.children[0].predict(X)
            Xb = self.children[1].predict(X)
            Xa_encode = []
            Xb_encode = []
            for xa,xb in zip(Xa,Xb):
                Xa_encode.append(encode(xa))
                Xb_encode.append(encode(xb))
            X_train = np.column_stack((X, Xa_encode))
            X_train = np.column_stack((X_train, Xb_encode))
            X_train = np.reshape(X_train,X_train.shape+(1,))
        for x in X_train:
            output = predict(self.NN,x)
            predictions.append(output)

        return predictions
        # check for more probability
    
X_train = load_mnist_images_to_df('./dataset/train-images.idx3-ubyte')
Y_train = load_mnist_labels('./dataset/train-labels.idx1-ubyte')
X_test = load_mnist_images_to_df('./dataset/t10k-images.idx3-ubyte')
Y_test = load_mnist_labels('./dataset/t10k-labels.idx1-ubyte')
X_train = X_train / 255.0
X_test = X_test / 255.0

from sklearn.utils import shuffle

def oversample(X_train,Y_train):
    X_train_df = pd.DataFrame(X_train)
    Y_train_series = pd.Series(Y_train)
    X_train_df = X_train_df.reset_index(drop=True)
    Y_train_series = Y_train_series.reset_index(drop=True)
    class_0 = X_train_df[Y_train_series == 0]
    class_1 = X_train_df[Y_train_series == 1]
    max_samples = max(len(class_0), len(class_1))
    class_0_upsampled = class_0.sample(max_samples, replace=True, random_state=42) if len(class_0) < max_samples else class_0
    class_1_upsampled = class_1.sample(max_samples, replace=True, random_state=42) if len(class_1) < max_samples else class_1
    X_train_balanced = pd.concat([class_0_upsampled, class_1_upsampled])
    Y_train_balanced = pd.Series([0] * max_samples + [1] * max_samples)
    X_train_balanced, Y_train_balanced = shuffle(X_train_balanced, Y_train_balanced, random_state=42)
    X_train_balanced = X_train_balanced.to_numpy()
    Y_train_balanced = Y_train_balanced.to_numpy()
    print(f"Balanced dataset shape: {X_train_balanced.shape}, {Y_train_balanced.shape}")
    return X_train_balanced,Y_train_balanced


X_val,X_train =  X_train.iloc[:2000], X_train.iloc[2000:]
Y_val,Y_train = Y_train[:2000],Y_train[2000:]
Y_train = np.where(Y_train == 0, 1, 0)
Y_val = np.where(Y_train == 0, 1, 0)
Y_test = np.where(Y_test == 0, 1, 0)
X_train,Y_train= oversample(X_train,Y_train)
X_val,Y_val= oversample(X_val,Y_val)


root = TreeNN(max_depth=3,parent_node=None)
root.train_tree(X_train,Y_train,X_val,Y_val)
pred = root.predict(X_test)
acc = 0
error = 0
for p,t in zip(pred,Y_test):
    if encode(p)==t:
        acc+=1
    error+=mse(p,t)
print("Test Accuracy = ",acc/len(Y_test))
print("Test Loss = ",error/len(Y_test))


data = root.NN[0].weights.squeeze()[:-2].reshape(28,28)
data = data*255
img = Image.fromarray(data,'L')
img.save('{i1}.png')
img.close()
