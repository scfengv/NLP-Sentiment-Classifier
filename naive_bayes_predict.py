import process_tweet


def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''

    word_l = process_tweet(tweet)

    p = 0

    p += logprior

    for word in word_l:
        if word in loglikelihood:

            p += loglikelihood[word]

    return p