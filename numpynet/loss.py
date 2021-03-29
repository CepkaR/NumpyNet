import numpy as np

class MSE:
    def __init__(self):
        self.name = "MSE"
    
    def __call__(self,y_true,y_pred):
        return np.mean(np.sum(np.power(y_true-y_pred, 2),axis=1, keepdims=True))
    
    def backward(self,y_true, y_pred):
        return 2*(y_pred-y_true)/len(y_true)
    
class Crossentropy:
    def __init__(self):
        self.name = "Crossentropy"
    
    def __call__(self,y_true,y_pred,eps = 1e-7):
        return -np.mean(np.sum(y_true*np.log(y_pred + eps),axis=1, keepdims=True))
    
    def backward(self,y_true, y_pred):
        # derivation with softmax
        return  (y_pred - y_true)/len(y_true)
