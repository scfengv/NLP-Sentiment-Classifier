# Sentiment Classifier with Logistic Regression and Naive Bayes

## Content
- Abstract
- Data Preprocessing
- Generative and Discriminative Classifiers
- Logistic Regression
    - The Cross-Entropy Loss Function
    - Gradient Descent
- Naive Bayes
- Reference

# Abstract


# Introduction
本文使用的是 `nltk` 中的 `twitter_samples` [資料集](https://www.nltk.org/howto/twitter.html)。共有 Positive Tweets 和 Negative Tweets 各 5000 筆真實 Twitter 平台上的資料，共 10000 筆。下面我將利用 Logistic Regression 中的 Sigmoid Function 和 Naive Bayes 兩種截然不同的分類方法試圖透過 Natural Laguage Processing 的方式做出 Tweets Sentiment Classifier。

# Data Preprocessing
```python
def process_tweet(tweet):
    """
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """

    # Cleaning
    tweet = re.sub(r'^RT[\s]+','',tweet)
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    tweet = re.sub(r'#', '',tweet)
    tweet = re.sub(r'@', '',tweet)

    # Tokenization
    token = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokenized = token.tokenize(tweet)

    # Stop words & Punctuation
    stopwords_english = stopwords.words('english')
    tweet_processed = []

    for word in tweet_tokenized:
        if (word not in stopwords_english and
        word not in string.punctuation):
            
            tweet_processed.append(word)
            
    # Stemming & Lowercasing
    tweet_stem = []
    stem = PorterStemmer()

    for word in tweet_processed:
        stem_word = stem.stem(word)
        tweet_stem.append(stem_word)
        
    return tweet_stem
```
文字清理的部分由四個步驟組成，分別是基本的 Cleaning, Tokenization, Remove Stop words and Punctuation 和 Stemming and Lowercasing。

**Cleaning** 移除 Twitter 中常見的如 RT (retweet), https, #, @

**Tokenization** 將句子拆成單詞

```python
tknzr = TweetTokenizer()
>>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
>>> tknzr.tokenize(s0)
['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3'
, 'and', 'some', 'arrows', '<', '>', '->', '<--']
```

**Remove Stop words & Punctuation** 移除 Tokenized 後在 `nltk.corpus.stopwords.words('english')` 和 `string.punctuation` 中的詞彙

**Stemming & Lowercasing** 將所有詞彙除去詞綴以得到詞根並全部變成小寫的過程。

$$
tun=
\begin{cases}
    \begin{matrix}
        tune\\
        tuned\\
        tuning
    \end{matrix}
\end{cases}
$$

```python
def build_freqs(tweets, ys):

    yslist = np.squeeze(ys).tolist()
    
    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):

            pair = (word, y)

            if pair in freqs:
                freqs[pair] += 1

            else:
                freqs[pair] = 1

    return freqs
```

***input***

`tweets`: a list of unprocessed tweets

`ys`: sentiment label (1, 0) of each tweet (m, 1)

***return***

`freqs`: a dictionary contains all of the words and it's sentiment frequency

`freqs.keys()`: (word, sentiment) (Ex: `('pleas', 1.0)`)

`freqs.values()`: frequency (Ex: `81`)

Which means that there are 81 positive tweets contain 'pleas'

```python
keys = ['''words that are interested''']

data = []

for word in keys:

    pos = 0
    neg = 0
    
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]
        
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]
        
    data.append([word, pos, neg])
    
data
```

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

在 **Logistic Regression** 模型中，首先將每句 Tweets 經過資料前處理後，組成一個擁有所有詞彙 (令有 m 個不同詞彙) 的 Vocabulary。
在計算特定詞彙在 Positive Tweets 和 Negative Tweets 中分別出現的次數後，每個詞彙會形成一個 (1x3) 的矩陣，分別是 bias、在 Postive 中的詞頻、在 Negative 中的詞頻 $[1\ (bias),\ pos,\ neg]$，所有詞彙彙整成一個矩陣會組成一個 (mx3) 的 input matrix (training X)。放入模型後模型會分別計算一個權重 $w_i$ 表示該 $x_i$ 對於分類的重要性 (這個 Case 中為 $w_1x_1\ (pos)$, $w_2x_2\ (neg)$ )。為了要達到成功預測，模型會計算一個 $z$ 值 (eq. 3) 以量化分類結果，此時 $z$ 會介於負無限到正無限之間 $z \in (-\infty, \infty)$ ，為了可以用機率的形式表示，將 $z$ 放入 Sigmoid Function (eq. 4) 後得到 $\sigma(z) \in (0,1)$ ，即可用來代表此字彙屬於 Positive sentiment 的機率 (Negative = 1 - $\sigma(z)$ ) 。Logistic Regression 做決策的過程如 eq. 5，若計算出來的 $\sigma(z)>0.5$，則會將其分類到 $y = 1$ (Positive sentiment)，反之則會被分類到 $y = 0$ (Negative sentiment)。

$$
z = (\sum_{i=1}^n w_ix_i)+b = w \cdot x+b \tag{3}
$$

$$
\sigma(z) = \frac{1}{1+\exp(-z)} = \frac{1}{1+\exp(-(w \cdot x + b))}\tag{4}
$$

$$
decision=
\begin{cases}
    \begin{matrix}
        1,\ if\ \sigma(z)>0.5\\
        0,\ if\ \sigma(z)<0.5
    \end{matrix}
\end{cases}\tag{5}
$$

```python
def sigmoid(z): 
    h = 1/(1 + np.exp(-z))    
    return h
```
<h3>The Cross-Entropy Loss Function</h3>

在 Logistic Regression 中用來量化模型表現的方式為去計算 Classifier output ( $\hat{y}=\sigma(w \cdot x +b)$ ) 和 Real output (y = 0 or 1, Bernoulli distribution) 之間的距離 (在 Gradient Descent 的部分有更詳細的說明)，稱為 Cost Function 或 Cross-Entropy Loss Function，在一個理想的情況下，一個完美的 Classifier 會 assign $P(y|x) = 1$ 給 $y = 1$，反之 $P(y|x) = 0$ 給 $y = 0$，藉由計算 $\hat{y}$ 和 y 之間的差距。
由於結果的二元離散分佈特性，故可以將模型做出正確決定的機率 $P(y|x)$ 表達為 eq. 6，將其取 Log 可得到更直觀的 eq. 7，以機率的觀念會希望求得 $\hat{y}$ 使 $\log(P(y|x))$ 最大，但在 Loss Function 的觀念中會希望越小越好，故將 $\log(P(y|x))$ 加一個負號 (eq. 8)，帶入 Sigmoid Function 即可得到 **Logistic Regression 中的 Loss Function** (eq. 9)。

$$
P(y|x) =  \hat{y}^y\ (1 - \hat{y})^{1-y}\tag{6}
$$

| P | y = 0 | y = 1
| --- | --- | --- |
| $\hat{y} = 0$ | 1 | 0 |
| $\hat{y} = 1$ | 0 | 1 |

$$
\tag{7} \log(P(y|x)) = y \log \hat{y} + (1-y) \log(1- \hat{y})
$$

$$
\tag{8} L(\hat{y}_i, y_i) = -\log(P(y|x)) = - [\ y \log \hat{y} + (1-y) \log(1- \hat{y})\ ]
$$

$$
\tag{9} L(\hat{y}_i, y_i) = -\log(P(y|x)) = - [\ y \log (\sigma(w \cdot x +b)) + (1-y) \log(1- \sigma(w \cdot x +b))\ ]
$$

若再將 Loss Function 做細部的拆解的話可以發現其實是由兩個部分所貢獻，分別是當 $y = 1$ 時主導的 $y \log \hat{y}$ 和 當 $y = 0$ 時主導的 $(1-y) \log(1- \hat{y})$，而兩部分的分析如下。

| y | $y \log \hat{y}$ | $(1-y) \log(1- \hat{y})$ |
| --- | --- | --- |
| 0 | 0 | any |
| 1 | any | 0 |


| y | $\hat{y}$ | $y \log \hat{y}$
| --- | --- | --- |
| 0 | any | 0 |
| 1 | 0.99 | ~0 |
| 1 | ~0 | $-\infty$ |

| y | $\hat{y}$ | $(1-y) \log(1- \hat{y})$ |
| --- | --- | --- |
| 1 | any | 0 |
| 0 | 0.01 | ~0 | 
| 0 | ~1 | $-\infty$ |


<h3>Gradient Descent</h3>

Gradient Descent 的目的是在找出一個理想的 $w_i$ 可以使 Loss Function 最小 (eq. 10)，以微積分的角度，梯度的方向是 Loss Function 在權重 $w_i$ 時的最大增加方向，量則是此方向上的增加量。

在做 Gradient Descent 的過程中，會先找出在權重 $w_i$ 下 Loss Function 的梯度方向，並嘗試往反方向移動，即可達到降低 Loss 的效果 (eq. 11)。由推導結果可以發現，梯度對於某 $i$ 變量的權重 $w_i$ 可以簡單表示為 Estimated $\hat{y}$ 和 True $y$ 之間的差距乘上 input value $x_i$。

Gradient Descent 的另一個參數 $\eta$ 是模型的 Learning rate，是一個需要被調整的超參數，可以視為模型在找到梯度方向後需要跨多大步，若 Learning rate 太大，會導致模型在 minimum Loss 的兩端來回遊走找不到最低點進而導致結果無法收斂 (稱為 Overshoot)，反之若 Learning rate 太小，則會導致模型學習速度太慢。常見的作法為先使用 High Learning rate 找到相對低點後，再將其慢慢降低。

$$
\tag{10} \hat{w} =  {\arg\min_{w}} \frac{1}{m} \sum_{i=1}^m L(f(w_i, x_i), y_i)
$$

$$
\tag{11} w^{t+1} = w^t - \eta \nabla L(f(w_i, x_i), y_i)
$$

$$
\nabla L(f(w_i, x_i), y_i) = 
\begin{bmatrix}
\frac{\partial}{\partial w_1}L\\
\frac{\partial}{\partial w_2}L\\
.\\
.\\
\frac{\partial}{\partial w_n}L\\
\end{bmatrix}
$$

$$
\frac{\partial}{\partial w_i}L = \frac{-\partial}{\partial w_i} [\ y \log (\sigma(w \cdot x +b)) + (1-y) \log(1- \sigma(w \cdot x +b))\ ]
$$

$$
\frac{\partial}{\partial w_i}L = -\frac{y}{\sigma(w \cdot x +b)} \frac{\partial}{\partial w_i} (\sigma (w \cdot x +b)) - \frac{(1-y)}{1- \sigma(w \cdot x +b)} \frac{\partial}{\partial w_i}(1- \sigma(w \cdot x +b))
$$

$$
\frac{\partial}{\partial w_i}L = -[\frac{y}{\sigma(w \cdot x +b)} - \frac{(1-y)}{1- \sigma(w \cdot x +b)}]\  \frac{\partial}{\partial w_i} (\sigma (w \cdot x +b))
$$

$$
After\ some\ complicate\ algebra...
$$

$$
\frac{\partial}{\partial w_i}L = [\sigma (w \cdot x +b)-y]x_i \\
= {\color{red}[\hat{y} - y] x_i}\ \ \ \ \ \ \ 
$$

```python
def gradientDescent(x, y, theta, alpha, num_iters):

    m = len(x)
    
    for i in range(0, num_iters):

        z = np.dot(x, theta)
        h = sigmoid(z)
        
        # Loss function
        J = -1/m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))

        # Gradient Descent
        theta = theta - alpha/m * (np.dot(x.T, (h - y)))
        
    J = float(J)

    return J, theta
```

***Input***

`x`: input matrix (m, n+1) (training x)

`y`: corresponging label matrix (m, 1) (training y)

`theta`: initial weight vector (n+1, 1)

`alpha`: learning rate

`num_iters`: max iteration number

***Return***

`J`: cost after training

`theta`: trained weight vector
    
<h3> Model Training </h3>

```python
X = np.zeros((len(train_x), 3))

Y = train_y

for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

t = []
for i in np.squeeze(theta):
    t.append(i)

print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {t}")
```
`The cost after training is 0.22480686`

`The resulting vector of weights is [6e-08, 0.00053854, -0.00055825]`


<h3> Model Testing </h3>

```python
def test_logistic_regression(test_x, test_y, freqs, theta):

    y_hat = []
    
    for tweet in test_x:

        x = extract_features(tweet, freqs)
        y_pred = sigmoid(np.dot(x, theta))
        
        if y_pred > 0.5:
            y_hat.append(1)

        else:
            y_hat.append(0)

    accuracy = (y_hat == np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy
```

`Logistic regression model's accuracy = 0.9950`

### Reference
[1] [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). Dan Jurafsky and James H. Martin Jan 7, 2023
