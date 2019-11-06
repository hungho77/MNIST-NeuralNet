import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import random
import math


class Neural_Net(object):
    def __init__(self, layers):
        self.dims = layers[0]
        self.activations = layers[1]
        self.params = {}
        self.L = len(self.dims)
        self.n = 0
        self.losses = []
        self.cache = {}


    def init_params(self):
        np.random.seed(1)

        for l in range(0, len(self.dims) - 1):
            self.params["W" + str(l+1)] = np.random.randn(self.dims[l+1], self.dims[l]) / np.sqrt(
                self.dims[l])
            self.params["b" + str(l+1)] = np.zeros((self.dims[l+1], 1))


    def relu(self, Z):
        return np.maximum(0,Z)


    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))


    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)


    def activation(self, Z, activations):
        if activations == 'sigmoid':
            A = self.sigmoid(Z)
        elif activations == 'relu':
            A = self.relu(Z)
        elif activations == 'softmax':
            A = self.softmax(Z)
        return A

    def forward(self, X):
        A = X.T

        for l in range(self.L - 1):
            # Compute output of hidden layers
            Z = self.params["W" + str(l + 1)].dot(A) + self.params["b" + str(l + 1)]
            A = self.activation(Z, self.activations[l])
            
            # Save result into cache
            self.cache["A" + str(l + 1)] = A
            self.cache["W" + str(l + 1)] = self.params["W" + str(l + 1)]
            self.cache["Z" + str(l + 1)] = Z

        # Compute last output of Neural Net
        Z = self.params["W" + str(self.L)].dot(A) + self.params["b" + str(self.L)]
        A = self.activation(Z, self.activations[self.L - 1])
        
        # Save result into cache
        self.cache["A" + str(self.L)] = A
        self.cache["W" + str(self.L)] = self.params["W" + str(self.L)]
        self.cache["Z" + str(self.L)] = Z

        return A


    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)


    def relu_derivative(self, Z):
        Z[Z<=0] = 0
        Z[Z>0] = 1
        return Z


    def activation_derivative(self, Z, activations):
        if activations == 'sigmoid':
            dZ = self.sigmoid_derivative(Z)
        elif activations == 'relu':
            dZ = self.relu_derivative(Z)
        elif activations == 'softmax':
            dZ = 1
        return dZ

    # Loss function cross-entropy
    def compute_loss(self, y, y_hat):
        return  -np.sum(y*np.log(y_hat.T))/self.n


    def backward(self, X, Y):
        derivatives = {}
        # Save input as A0 in cache
        self.cache["A0"] = X.T

        # Load last output's result from cache
        A = self.cache["A" + str(self.L)]

        # Compute derivative to backpropagation
        dZ = (A - Y.T)*(self.activation_derivative(self.cache["Z" + str(self.L)], self.activations[self.L - 1])) / self.n
        dW = dZ.dot(self.cache["A" + str(self.L - 1)].T)
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.cache["W" + str(self.L)].T.dot(dZ) 

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        # Compute derivative
        for l in range(self.L - 1, 0, -1):
            dZ = dA_prev * self.activation_derivative(self.cache["Z" + str(l)], self.activations[ l - 1])
            dW = dZ.dot(self.cache["A" + str(l - 1)].T)
            db = np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = self.cache["W" + str(l)].T.dot(dZ)
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives


    def update_weight(self, derivatives, lr):
 
        # update weight
        for l in range(1, self.L + 1):
            self.params["W" + str(l)] = self.params["W" + str(l)] - lr * derivatives[
                "dW" + str(l)]
            self.params["b" + str(l)] = self.params["b" + str(l)] - lr * derivatives[
                "db" + str(l)]


    def random_mini_batches(self,X,y,mini_batch_size):
        m = X.shape[0]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation,:]
        shuffled_Y = y[permutation,:]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
 
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[ k * mini_batch_size: (k + 1) * mini_batch_size,:]
            mini_batch_Y = shuffled_Y[ k * mini_batch_size: (k + 1) * mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


    def fit(self, X, Y, lr=0.01, epochs=20, batch_size=32):
        np.random.seed(1)

        # Hyper parameters
        self.n = batch_size
        self.dims.insert(0, X.shape[1])

        # Init parameters of the network
        self.init_params()

        for e in range(epochs):
            # Load batch
            mini_batches = self.random_mini_batches(X, Y, batch_size)

            for mini_batch_X, mini_batch_Y in mini_batches:
                A = self.forward(mini_batch_X)
                loss = self.compute_loss(mini_batch_Y, A)
                derivatives = self.backward(mini_batch_X, mini_batch_Y)
                self.update_weight(derivatives, lr)

            self.losses.append(loss)
            print("Epochs %d/%d - loss: %.5f - acc: %.5f" % (e+1, epochs, loss, self.evaluate(X, Y)))


    def predict(self, X):
        A = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        return y_hat


    def evaluate(self, X, Y):
        y_hat = self.predict(X)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100


    def plot_loss(self):
        plt.figure()
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()


def pre_process_data(X_train, y_train, X_val, y_val, X_test):
    enc = OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1))
    y_val = enc.transform(y_val.reshape(len(y_val), -1))

    return X_train, y_train, X_val, y_val, X_test


def load_data(path):
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')

    df_features = df_train.iloc[:, 1:785]
    df_label = df_train.iloc[:, 0]
    X_test = df_test.iloc[:, 0:784]

    X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label, 
                                                test_size = 0.2,
                                                random_state = 1212)

    X_train = np.array(X_train).reshape(33600, 784)
    X_cv = np.array(X_cv).reshape(8400, 784)
    X_test = np.array(X_test).reshape(28000, 784)
    
    # Feature Normalization 
    X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')
    X_train /= 255; X_cv /= 255; X_test /= 255
    y_train = np.array(y_train)
    y_cv = np.array(y_cv)
    
    return X_train, y_train, X_cv, y_cv, X_test


if __name__ == '__main__':
    # Call argument for command
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='/home/hung/study/SOFT_COMPUTING/NN_lab01/digit-recognizer/', type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    # load data
    X_train, y_train, X_val, y_val, X_test = load_data(args.path)

    # preprocess data
    X_train, y_train, X_val, y_val, X_test = pre_process_data(X_train, y_train, X_val, y_val, X_test)

    print("X_train's shape: " + str(X_train.shape))
    print("X_val's shape: " + str(X_val.shape))

    # Build neural network
    layers_dims = [64, 10]
    act_fucntion = ['relu', 'softmax']
    layers = (layers_dims, act_fucntion)
    net = Neural_Net(layers)

    # Train network
    net.fit(X_train, y_train, args.lr, args.epochs, args.batch_size)
    print("Train Accuracy:", net.evaluate(X_train, y_train))
    print("Test Accuracy:", net.evaluate(X_val, y_val))

    # Predict with test data
    y_hat = net.predict(X_test)

    # Write y_hat to csv file to submit
    df = pd.DataFrame({'ImageId': np.arange(1, y_hat.shape[0] + 1), 'Label': y_hat[:]})
    export_csv = df.to_csv (r'sample_submission.csv', index = None, header=True)