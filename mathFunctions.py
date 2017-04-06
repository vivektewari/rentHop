import numpy as np
def sigmoid(x):
    z=x
    #if np.isnan(z).any():
    return np.exp(z)/(1+np.exp(z))


def sigDeriv(x):return sigmoid(x)*(1-sigmoid(x))