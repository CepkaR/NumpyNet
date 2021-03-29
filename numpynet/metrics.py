import numpy as np

def acc(y_true,y_pred):
    # input must by onehot_en
    return  np.mean(np.argmax(y_pred,axis=1) == np.argmax(y_true,axis=1))
