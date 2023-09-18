# Sentiment Classifier with Logistic Regression and Naive Bayes

## Content

# Abstract


# Introduction
本文使用的是 `nltk` 中的 `twitter_samples` [資料集](https://www.nltk.org/howto/twitter.html)。共有 Positive Tweets 和 Negative Tweets 各 5000 筆真實 Twitter 平台上的資料，共 10000 筆。下面我將利用 Logistic Regression 中的 Sigmoid Function 和 Naive Bayes 兩種截然不同的分類方法試圖透過 Natural Laguage Processing 的方式做出 Tweets Sentiment Classifier。


# Logistic Regression
在 **Logistic Regression** 模型中，首先將每句 Tweets 經過資料前處理 (移除 Stop words, Punctuation, @, #, URLs 和 Stemming) 過後，組成一個擁有所有詞彙 (令有 m 個不同詞彙) 的 Vocabulary。在計算特定詞彙在 Positive Tweets 和 Negative Tweets 中分別出現的次數後，每個詞彙會形成一個 (1x3) 的矩陣，分別是 bias、在 Postive 中的詞頻、在 Negative 中的詞頻 [1 (bias), pos, neg]，所有詞彙彙整成一個矩陣會組成一個 (mx3) 的 input matrix (training X)，再放入 Sigmoid Function 中，


