import numpy as np
from numpynet.function import relu,relu_der, tanh,tanh_der, sigmoid,sigmoid_der, softmax,softmax_der 

class Activation:
    def __init__(self, activation="relu",name="activation"):
        self.name = name
        self._param_bool = False
        
        if activation == "relu":
            self.fun = relu
            self.fun_der = relu_der
            
        elif activation == "tanh":
            self.fun = tanh
            self.fun_der = tanh_der
            
        elif activation == "sigmoid":
            self.fun = sigmoid
            self.fun_der = sigmoid_der
        
        elif activation == "softmax":
            self.fun = softmax
            self.fun_der = softmax_der
        
        else:
            raise ValueError('Name of activation is invalid')
            
        
    def forward(self, X):
        self.X = X
        self.Y = self.fun(X)
        return self.Y

    def backward(self, Y_er):
        return self.fun_der(self.X)*Y_er, None, None

       
    

class Dense:
    def __init__(self, input_size, output_size, initialization="uniform",name="dense"):
        self.name = name
        self._param_bool = True
        
        if initialization == "uniform":
            self.W = 2*np.random.rand(input_size, output_size) - 1
            self.b = 2*np.random.rand(1, output_size) - 1
        elif initialization == "he":
            std = np.sqrt(2.0 / input_size)
            self.W = np.random.normal(scale = std,
                                      size= (input_size, output_size))
            
            self.b =  np.random.normal(scale = std,
                                      size= (1, output_size))
        elif initialization == "xavier":
            low , high = -(1.0 / np.sqrt(input_size)), (1.0 / np.sqrt(input_size))
            self.W = np.random.uniform(low=low, high=high,
                                       size= (input_size, output_size))
            self.b = np.random.uniform(low=low, high=high,
                                       size= (1, output_size))
         

    def forward(self, X):
        self.X = X
        self.Y = np.dot(self.X, self.W) + self.b
        return self.Y

    def backward(self, Y_er):
        X_er = np.dot(Y_er, self.W.T)
        W_er = np.dot(self.X.T, Y_er)
        b_er = np.sum(Y_er,axis=0, keepdims=True)
        return X_er,W_er,b_er
    
    