import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """

    # cleaning
    tweet = re.sub(r'^RT[\s]+','',tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '',tweet)
    tweet = re.sub(r'@', '',tweet)

    # tokenization
    token = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokenized = token.tokenize(tweet)

    # STOP WORDS
    stopwords_english = stopwords.words('english')
    tweet_processed = []

    for word in tweet_tokenized:
        if (word not in stopwords_english and
        word not in string.punctuation):
            
            tweet_processed.append(word)
            
    # stemming 
    tweet_stem = []
    stem = PorterStemmer()

    for word in tweet_processed:
        stem_word = stem.stem(word)
        tweet_stem.append(stem_word)
        
    return tweet_stem