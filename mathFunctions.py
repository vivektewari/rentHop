import numpy as np
def sigmoid(x):
    z=x
    if np.isnan(z).any():
        print z
        y=0
    z[z>600.0]=600
    z[z<-600.0]=-600.0
    return np.exp(z)/(1+np.exp(z))


def sigDeriv(x):return sigmoid(x)*(1-sigmoid(x))