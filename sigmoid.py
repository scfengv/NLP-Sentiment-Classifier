import numpy as np


def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    h = 1/(1+np.exp(-z))
    
    return h
