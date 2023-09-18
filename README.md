# Sentiment Classifier with Logistic Regression and Naive Bayes

## Content

# Abstract


# Introduction
本文使用的是 `nltk` 中的 `twitter_samples` [資料集](https://www.nltk.org/howto/twitter.html)。共有 Positive Tweets 和 Negative Tweets 各 5000 筆真實 Twitter 平台上的資料，共 10000 筆。下面我將利用 Logistic Regression 中的 Sigmoid Function 和 Naive Bayes 兩種截然不同的分類方法試圖透過 Natural Laguage Processing 的方式做出 Tweets Sentiment Classifier。

# Generative and Discriminative Classifiers
Logistic Regression 和 Naive Bayes 兩者最大的不同在於 Losgistic Regression 屬於 Discriminative Classifiers 而 Naive Bayes 屬於 Generative Classifiers <sub>[1]</sub>。以貓狗分類器作為例子，Generative Classifier 會想去了解貓和狗分別長什麼樣子，而 Discriminative Classifier 只想知道該怎麼去分辨這兩個種類。也可以透過 eq. 1 & 2 來了解兩者的差異，在 Generative Classifier 中，模型會基於 $\color{red}Prior$ 對於資料機率分布的假設 (Ex: Gaussian, Laplacian) 去計算類別的 $\color{blue}Likelihood$，但在 Discriminative Classifier 中，模型則是直接計算事後機率 $P(c|d)$，對資料不進行任何假設。在計算上 Discriminative Classifier 是透過 Gradient Descent 直接將 $w\ (weight),\ b\ (bias)$ 找出，而 Generative Classifier 會利用 Singular Value Decomposition (SVD) 先找出 $U\ (unitary\ matrix)$, $\Sigma\ (rectangular\ matrix)$, $V^*\ (conjugate\ transpose)$，再計算出 $w\ (weight),\ b\ (bias)$。

$$
P(c|d) = P(d|c)*\frac{P(c)}{P(d)} \tag{1}
$$

$$
\hat{c} = \arg\max_{c} {\color{blue}P(d|c)}*{\color{red}P(c)} \tag{2}
$$

$$
{\color{blue}P(d|c)}:Likelihood\ \ \ \ \ {\color{red}P(c)}:Prior
$$

# Logistic Regression
在 NLP 中 Logistic Regression 是一個很基礎的監督式學習分類演算法，神經網路即是由一系列的 Logistic Regression Classifiers 堆疊而成的。
在 **Logistic Regression** 模型中，首先將每句 Tweets 經過資料前處理 (移除 Stop words, Punctuation, @, #, URLs 和 Stemming) 過後，組成一個擁有所有詞彙 (令有 m 個不同詞彙) 的 Vocabulary。
在計算特定詞彙在 Positive Tweets 和 Negative Tweets 中分別出現的次數後，每個詞彙會形成一個 (1x3) 的矩陣，分別是 bias、在 Postive 中的詞頻、在 Negative 中的詞頻 $[1\ (bias),\ pos,\ neg]$，所有詞彙彙整成一個矩陣會組成一個 (mx3) 的 input matrix (training X)，再放入 Sigmoid Function 中，


### Reference
[1] [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). Dan Jurafsky and James H. Martin Jan 7, 2023