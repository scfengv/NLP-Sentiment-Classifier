import sigmoid
import numpy as np
import extract_features


def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''

    x = extract_features(tweet, freqs)

    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred