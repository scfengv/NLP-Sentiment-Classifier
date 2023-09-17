import sigmoid
import numpy as np


def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    '''

    m = len(x)
    
    for i in range(0, num_iters):

        z = np.dot(x, theta)
        
        h = sigmoid(z)
        
        J = -1/m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

        theta = theta - alpha/m * (np.dot(x.T, (h - y)))
        
    J = float(J)
    return J, theta