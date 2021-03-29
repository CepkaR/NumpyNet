import numpy as np

def shuffle_data(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]

def train_test_split(X,y,train_size=None,shuffle=False):
    if shuffle:
        X,y = shuffle_data(X,y)
    split = int(len(X)*train_size)
    return X[:split,:], y[:split,:],X[split:,:], y[split:,:]

class StandartScaler:
    def __init__(self):
        self.u = None
        self.s = None
        
    def fit(self,X):
        self.u = np.mean(X,axis=0)
        self.s = np.std(X,axis=0)
        return self
    
    def transform(self,X):
        return (X - self.u)/self.s
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

class Dataloader:
    def __init__(self,X,y,batch_size,shuffle=False):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.i = -1
        self.stop = False
        self.shuffle = shuffle
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.i == -1:
            self.X, self.y = shuffle_data(self.X,self.y)
        self.i += 1
        if self.stop:
            self.i = -1
            self.stop = False
            raise StopIteration
            
        if len(self.y) <= (self.i+1)*self.batch_size:
            X_bach = self.X[self.i*self.batch_size:,:]
            y_bach = self.y[self.i*self.batch_size:,:]
            self.stop = True
            return X_bach, y_bach 
            
        X_bach = self.X[self.i*self.batch_size:(self.i+1)*self.batch_size,:]
        y_bach = self.y[self.i*self.batch_size:(self.i+1)*self.batch_size,:]
        return X_bach, y_bach 

    
    def __len__(self):
        return int(np.ceil(self.X.shape[0]/self.batch_size))

