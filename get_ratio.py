import lookup


def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words
        word: string to lookup

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}

    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    pos_neg_ratio['negative'] = lookup(freqs, word, 0)

    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1)/(pos_neg_ratio['negative'] + 1)
    
    return pos_neg_ratio