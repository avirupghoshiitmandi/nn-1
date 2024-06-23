import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

# Reading the data
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Converting to NumPy arrays
data = np.array(data)
test_data = np.array(test_data)

# Getting shapes
m, n = data.shape
j, k = test_data.shape

# Transposing the data
data = data.T
test_data = test_data.T

# Separating features and labels
X_data = data[1:]  # All rows except the first
Y_data = data[0]   # Only the first row

# Initialize parameters
def input_parameters():
    w1 = np.random.randn(20, 784) * 0.01
    b1 = np.zeros((20, 1))
    w2 = np.random.randn(20, 20) * 0.01
    b2 = np.zeros((20, 1))
    w3 = np.random.randn(10, 20) * 0.01
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3
w1_F,b1_F,w2_F,b2_F,w3_F,b3_F=input_parameters()
# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward propagation
def forward_prop(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    A1 = ReLU(z1)
    A1_MAX= np.argmax(A1, 0)
    z2 = w2.dot(A1) + b2
    A2 = ReLU(z2)
    A2_MAX = np.argmax(A2,0)
    z3 = w3.dot(A2) + b3
    A3 = softmax(z3)
    A3_MAX = np.argmax(A3,0)
    return z1, A1, z2, A2, z3, A3

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Derivative of ReLU
def ReLU_derivative(Z):
    return Z > 0

# Backward propagation
def backward_prop(z1, A1, z2, A2, z3, A3, Y, w3, X, w2):
    one_hot_Y = one_hot(Y)
    dz3 = A3 - one_hot_Y
    dw3 = (1 / m) * dz3.dot(A2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)
    dz2 = w3.T.dot(dz3) * ReLU_derivative(z2)
    dw2 = (1 / m) * dz2.dot(A1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * ReLU_derivative(z1)
    dw1 = (1 / m) * dz1.dot(X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2, dw3, db3
def save_parameters(W1, B1, W2, B2, W3, B3, folder_path='./parameters'):
    os.makedirs(folder_path, exist_ok=True)
    np.save(os.path.join(folder_path, 'W1.npy'), W1)
    np.save(os.path.join(folder_path, 'B1.npy'), B1)
    np.save(os.path.join(folder_path, 'W2.npy'), W2)
    np.save(os.path.join(folder_path, 'B2.npy'), B2)
    np.save(os.path.join(folder_path, 'W3.npy'), W3)
    np.save(os.path.join(folder_path, 'B3.npy'), B3)
# Updating parameters
def update(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3

# Accuracy function
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Get predictions
def get_predictions(A3):
    return np.argmax(A3, 0)

# Gradient descent
def gradient_descent(X, Y, iterations, alpha):
    global w1_F, b1_F, w2_F, b2_F, w3_F, b3_F
    max=0
    W1, B1, W2, B2, W3, B3 = input_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, B1, W2, B2, W3, B3, X)
        dW1, dB1, dW2, dB2, dW3, dB3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, Y, W3, X, W2)
        W1, B1, W2, B2, W3, B3 = update(W1, B1, W2, B2, W3, B3, dW1, dB1, dW2, dB2, dW3, dB3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            acc=get_accuracy(predictions, Y)
            
            print("Accuracy:",acc )
            if acc>max:
                w1_F,b1_F,w2_F,b2_F,w3_F,b3_F=W1, B1, W2, B2, W3, B3
                max=acc
    return W1, B1, W2, B2, W3, B3

# Example usage:
iterations = 800
alpha = 0.01
W1, B1, W2, B2, W3, B3 = gradient_descent(X_data, Y_data, iterations, alpha)
def make_predictions( W1, b1, W2, b2,W3,b3,X):
    _,_,_, _, _, A3 = forward_prop(W1, b1, W2, b2,W3,b3, X)
    predictions = get_predictions(A3)
    return predictions

def prediction(index, W1, b1, W2, b2,W3,b3):
    current_image = test_data[:, index, None]
    prediction = make_predictions(W1, b1, W2, b2,W3,b3,test_data[:, index, None])
    print("Prediction: ", prediction)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

save_parameters(w1_F, b1_F, w2_F, b2_F, w3_F, b3_F,"PATH/OF/YOUR/REPO/nn-1")
            
