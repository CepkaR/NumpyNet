import numpy as np

def relu(x):
    return np.maximum(x,0)

def relu_der(x):
    return (x>1).astype(int)

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1 - sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
   
def softmax_der(x):
    return softmax(x)*(1-softmax(x))
