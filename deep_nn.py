"""
    Deep neural networks for binary classification.
    (An object is "xyz" or not.)
"""

# pylint: disable = line-too-long, too-many-lines, too-many-arguments, wrong-import-order, invalid-name, missing-docstring

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

def getDataset(path, pixel):
    """
    A function which reads images and resize them.
    - pixel: width and height of the images after resizing
    """
    images = np.empty((pixel*pixel*3, 0)) # numpy array of flatten images, each column is an image
    classes = np.empty((1, 0)) # binary classification, 0 or 1

    # getting images:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                imagePath = os.path.join(root, file)
                image = np.array(plt.imread(imagePath)) # reading the image and transforming it to a numpy array
                image = image if len(image.shape) == 3 else color.gray2rgb(image) # if it's grayscaled, transform it to rgb
                # resizing and flattening the image:
                image = np.array(Image.fromarray(image.astype('uint8')).resize((pixel, pixel))).reshape(1, -1).T
                image = image/255 # normalizing
                images = np.append(images, image, axis=1)
                classes = np.append(classes, 1 if file[:3] == "air" else 0) # classifying the image as "airplane" or "non-airplane"

    return images, classes.reshape(1, -1)

#######################

def saveDataset(X_train, Y_train, X_test, Y_test, path):
    """
    A function which saves numpy arrays of a dataset to binary files.
    """
    np.save(os.path.join(path, "X_train.npy"), X_train)
    np.save(os.path.join(path, "Y_train.npy"), Y_train)
    np.save(os.path.join(path, "X_test.npy"), X_test)
    np.save(os.path.join(path, "Y_test.npy"), Y_test)

#######################

def loadDataset(path):
    """
    A function which gets numpy arrays of a dataset from binary files.
    """
    X_train = np.load(os.path.join(path, "X_train.npy"))
    Y_train = np.load(os.path.join(path, "Y_train.npy")).reshape(1, -1)
    X_test = np.load(os.path.join(path, "X_test.npy"))
    Y_test = np.load(os.path.join(path, "Y_test.npy")).reshape(1, -1)

    return X_train, Y_train, X_test, Y_test

#######################

def sigmoid(z): # sigmoid activation function
    return 1 / (1 + np.exp(-z))

#######################

def sigmoid_derivative(z): # derivation of sigmoid activation function
    s = sigmoid(z)
    return s * (1-s)

#######################

def relu(z): # ReLU activation function
    return z * (z >= 0)

#######################

def relu_derivative(z): # derivation of ReLU activation function
    return 1 * (z >= 0)

#######################

def initializeValues(layer_sizes):
    """
    A function which initializes weights and biases of the neural network.
    - layer_sizes: list of # of hidden units for each layer
    """
    W, b = {}, {} # weights and biases, resceptively

    for l in range(1, len(layer_sizes)):
        W[l] = np.random.randn(layer_sizes[l], layer_sizes[l-1])*0.01
        b[l] = np.zeros((layer_sizes[l], 1))

    return W, b

#######################

def neuralNetwork(W, b, X, Y, m, L, alpha=0.05, num_of_iterations=1000, printCost=False):
    """
    The function of neural network model.
    ReLU activation function for layers 1 to L-1, and sigmoid activation function for the last layer.

    W: list of weights for each layer
    b: list of biases for each layer
    X: images
    Y: classes of images
    m: # of images
    L: # of layers
    alpha: learning rate
    """
    for i in range(num_of_iterations):
        # forward prop:
        Z, A = {}, {}
        A[0] = X

        for l in range(1, L-1):
            Z[l] = np.dot(W[l], A[l-1]) + b[l]
            A[l] = relu(Z[l])

        Z[L-1] = np.dot(W[L-1], A[L-2]) + b[L-1]
        A[L-1] = sigmoid(Z[L-1])

        # computing cost:
        J = -(np.dot(Y, np.log(A[L-1]).T) + np.dot((1-Y), np.log(1-A[L-1]).T)) / m

        if printCost and i % 100 == 0:
            print("Cost for iter " + str(i) + ": " + str(np.squeeze(J)))

        # backward prop:
        dA = -Y/A[L-1] + (1-Y)/(1-A[L-1])
        dW, dB = {}, {}

        dZ = dA * sigmoid_derivative(Z[L-1])
        dW[L-1] = np.dot(dZ, A[L-2].T) / m
        dB[L-1] = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W[L-1].T, dZ)

        for l in reversed(range(1, L-1)):
            dZ = dA * relu_derivative(Z[l])
            dW[l] = np.dot(dZ, A[l-1].T) / m
            dB[l] = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.dot(W[l].T, dZ)

        # gradient descent:
        for l in range(1, L):
            W[l] -= alpha*dW[l]
            b[l] -= alpha*dB[l]

#######################

def predict(W, b, L, X):
    """
    A function which predicts if an image is "xyz" or not.
    """
    A = X

    for l in range(1, L-1):
        Z = np.dot(W[l], A) + b[l]
        A = relu(Z)

    Z = np.dot(W[L-1], A) + b[L-1]
    A = sigmoid(Z)

    return np.round(A)

#######################

def main():
    print("Loading dataset...")

    """
    X_train, Y_train = getDataset("dataset/train", 32)
    X_test, Y_test = getDataset("dataset/test", 32)
    saveDataset(X_train, Y_train, X_test, Y_test, "npy")
    """

    X_train, Y_train, X_test, Y_test = loadDataset("npy")

    print("Dataset dimensions:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # this model example has 3 layers:
    # first layer has pixels of an image
    # second layer has 7 hidden units
    # last layer has 1 unit, which represents a binary classification
    # (you can change # of layers and theirs # of hidden units, except first and last layers)
    
    layer_sizes = [X_train.shape[0], 7, Y_train.shape[0]]
    L = len(layer_sizes)
    W, b = initializeValues(layer_sizes)
    m = X_train.shape[1]

    neuralNetwork(W, b, X_train, Y_train, m, L, alpha=0.01, num_of_iterations=1000, printCost=True)

    y_prediction_train = predict(W, b, L, X_train)
    y_prediction_test = predict(W, b, L, X_test)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

if __name__ == "__main__":
    main()
