import numpy as np

class SGD:
    def __init__(self,lr=0.001):
        self.lr = lr
        
        
    def update(self,W,b,dW,db):

        W -= self.lr*dW
        b -= self.lr*db
        
        return W,b
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class Adam:
    def __init__(self,lr=0.001,beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_b = 0
        self.v_b = 0
        self.m_W = 0
        self.v_W = 0
        self.t = 0
        
        
    def update(self,W,b,dW,db):
        self.t+=1
        
        self.m_W = self.beta1*self.m_W + (1-self.beta1)*dW
        self.m_b = self.beta1*self.m_b + (1-self.beta1)*db
        
        
        self.v_W = self.beta2*self.v_W + (1-self.beta2)*(dW**2)
        self.v_b = self.beta2*self.v_b + (1-self.beta2)*(db**2)
        
        self.m_hat_W = self.m_W/(1-(self.beta1**self.t))
        self.m_hat_b = self.m_b/(1-(self.beta1**self.t))
        
        self.v_hat_W = self.v_W/(1-(self.beta2**self.t))
        self.v_hat_b = self.v_b/(1-(self.beta2**self.t))
        
       
        W -= self.lr*self.m_hat_W/(np.sqrt(self.v_hat_W) + self.eps)
        b -= self.lr*self.m_hat_b/(np.sqrt(self.v_hat_b) + self.eps)
        
        return W,b
        
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    
class LR_scheduler:
    def __init__(self,lr=1e-3,lr_end=1e-8, mode="min",factor=0.1,patience=10,threshold=1e-4):
        self.lr = lr
        self.lr_end = lr_end
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.treshold = threshold
        self.best_metric = None
        self.num_bad = 0
        
    def step(self,metric):
        if self.best_metric == None:
            self.best_metric = metric
            
        if self._is_better(metric,self.best_metric):
            self.best_metric = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        
        if self.num_bad >= self.patience:
            self.lr *= self.factor
            self.lr = np.round(self.lr,decimals=20)
            self.num_bad = 0
        if self.lr <= self.lr_end:
            return "stop"
        return self.lr
        
        
    def _is_better(self,current,best):
        if self.mode == "min":
            return current < best - self.treshold
        
        elif self.mode == "max":
            return current > best + self.treshold
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
        
        
