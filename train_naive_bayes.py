import lookup
import numpy as np


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()]) # A set is a collection of unique data
    V = len(vocab)

    N_pos = N_neg = 0

    for pair in freqs.keys():

        if pair[1] > 0:
            N_pos += freqs[pair]

        else:
            N_neg += freqs[pair]

    D = len(train_y)

    D_pos = len(list(filter(lambda x: x > 0, train_y)))

    D_neg = len(list(filter(lambda x: x <= 0, train_y)))

    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:

        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)

        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

    return logprior, loglikelihood