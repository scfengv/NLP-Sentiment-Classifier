# Sentiment Classifier with Logistic Regression and Naive Bayes

## Content
- Abstract
- Data Preprocessing
- Generative and Discriminative Classifiers
- Logistic Regression
    - The Cross-Entropy Loss Function
    - Gradient Descent
- Naive Bayes
    - Conditional Probability
- Reference

# Abstract
本文將探討 Discriminative 和 Generative Algorithms 作為 Natural Language Sentiment Classifier 時的原理和分類模式，將用一點點的代數、機率和微積分加以佐證，內容包括了 Maximum Likelihood Estimation, Cross-Entropy Loss Function, Gradient Descent 和 Conditional Probability。

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
{('followfriday', 1.0): 23,
 ('france_int', 1.0): 1,
 ('pkuchli', 1.0): 1,
 ('57', 1.0): 2,
 ('milipol_pari', 1.0): 1,
 ('top', 1.0): 30,
 ('engag', 1.0): 7,
 ('member', 1.0): 14,
 ('commun', 1.0): 27,
 ('week', 1.0): 72,
 (':)', 1.0): 2960,
 ('lamb', 1.0): 1,
 ('2ja', 1.0): 1,
 ('hey', 1.0): 60,
 ('jame', 1.0): 7,
 ('odd', 1.0): 2,
 (':/', 1.0): 5,
 ('pleas', 1.0): 81}
 ```


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

<img width="700" src="https://github.com/scfengv/NLP-Sentiment-Classifier/assets/123567363/fa9c8c2f-9eef-4853-a1c6-01c749816ce5">


# Generative and Discriminative Classifiers
Logistic Regression 和 Naive Bayes 兩者最大的不同在於 Losgistic Regression 屬於 Discriminative Classifiers 而 Naive Bayes 屬於 Generative Classifiers <sub>[1]</sub>。
Discriminative Classifier 著重在分類，在畫出兩個類別中間的邊界線，因此不像 Generative Classifier 會對資料做假設並計算條件機率。Generative Classifier 顧名思義在找出如何可以生成類似於訓練集的資料新資料點的模型，因此更著重於訓練集中類別的資料分布，學習分佈的特性及型態。
以貓狗分類器作為例子，Generative Classifier 會想去了解貓和狗分別長什麼樣子，而 Discriminative Classifier 只想知道該怎麼去分辨這兩個種類。也可以透過 eq. 1 & 2 來了解兩者的差異，在 Generative Classifier 中，模型會基於 $\color{red}Prior$ 對於資料機率分布的假設 (Ex: Gaussian, Laplacian) 去計算類別的 $\color{blue}Likelihood$ ，但在 Discriminative Classifier 中，模型則是直接計算事後機率 $P(c|d)$ ，對資料不進行任何假設。在計算上 Discriminative Classifier 是透過 Gradient Descent 直接將 $w\ (weight),\ b\ (bias)$ 找出，而 Generative Classifier 會利用 Singular Value Decomposition (SVD) 先找出 $U\ (unitary\ matrix)$, $\Sigma\ (rectangular\ matrix)$ , $V^*\ (conjugate\ transpose)$，再計算出 $w\ (weight),\ b\ (bias)$ 。

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
\frac{\partial}{\partial w_i}L = [\sigma (w \cdot x +b)-y]x_i
$$

$$
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
    

<h3> Tuning Parameter </h3>

```python
X = np.zeros((len(train_x), 3))

Y = train_y
```

```python
alpha_values = [1e-9, 1e-10, 1e-11, 1e-12]
num_iters_values = [1000, 5000, 10000, 50000, 1000000]

cost_values = np.empty((len(alpha_values), len(num_iters_values)))

for i, alpha in enumerate(alpha_values):
    for j, num_iters in enumerate(num_iters_values):

        start_time = time.time()
        J, _ = gradientDescent(X, Y, np.zeros((3, 1)), alpha, int(num_iters))

        end_time = time.time()
        time_consume = end_time - start_time

        cost_values[i, j] = J
        print(f'alpha = {alpha}, iter = {num_iters} Calculated, Cost = {J:.4f}, Time elapsed: {time_consume:.2f} sec')
    
    print('---------------------------------------------')

plt.figure(figsize = (10, 6))
contour = plt.contourf(np.log10(alpha_values), num_iters_values, cost_values, levels = 20, cmap = 'viridis')
plt.colorbar(contour, label = 'Cost (J)')

plt.xlabel('log10(Learning Rate alpha)')
plt.ylabel('Number of Iterations (num_iters)')

plt.title('Cost vs. Learning Rate and Number of Iterations')

plt.show()
```

```python
alpha = 1e-09, iter = 1000 Calculated, Cost = 0.2773, Time elapsed: 0.63 sec
alpha = 1e-09, iter = 5000 Calculated, Cost = 0.1286, Time elapsed: 2.89 sec
alpha = 1e-09, iter = 10000 Calculated, Cost = 0.1013, Time elapsed: 5.81 sec
alpha = 1e-09, iter = 50000 Calculated, Cost = nan, Time elapsed: 34.93 sec
alpha = 1e-09, iter = 1000000 Calculated, Cost = nan, Time elapsed: 670.09 sec
---------------------------------------------
alpha = 1e-10, iter = 1000 Calculated, Cost = 0.5952, Time elapsed: 0.58 sec
alpha = 1e-10, iter = 5000 Calculated, Cost = 0.3847, Time elapsed: 2.94 sec
alpha = 1e-10, iter = 10000 Calculated, Cost = 0.2773, Time elapsed: 7.85 sec
alpha = 1e-10, iter = 50000 Calculated, Cost = 0.1286, Time elapsed: 33.51 sec
alpha = 1e-10, iter = 1000000 Calculated, Cost = nan, Time elapsed: 700.17 sec
---------------------------------------------
alpha = 1e-11, iter = 1000 Calculated, Cost = 0.6820, Time elapsed: 0.62 sec
alpha = 1e-11, iter = 5000 Calculated, Cost = 0.6404, Time elapsed: 2.93 sec
alpha = 1e-11, iter = 10000 Calculated, Cost = 0.5951, Time elapsed: 7.89 sec
alpha = 1e-11, iter = 50000 Calculated, Cost = 0.3847, Time elapsed: 33.50 sec
alpha = 1e-11, iter = 1000000 Calculated, Cost = 0.1013, Time elapsed: 682.91 sec
---------------------------------------------
alpha = 1e-12, iter = 1000 Calculated, Cost = 0.6920, Time elapsed: 0.57 sec
alpha = 1e-12, iter = 5000 Calculated, Cost = 0.6875, Time elapsed: 3.23 sec
alpha = 1e-12, iter = 10000 Calculated, Cost = 0.6819, Time elapsed: 7.58 sec
alpha = 1e-12, iter = 50000 Calculated, Cost = 0.6404, Time elapsed: 33.33 sec
alpha = 1e-12, iter = 1000000 Calculated, Cost = 0.2772, Time elapsed: 683.42 sec
```

![下載 (1)](https://github.com/scfengv/NLP-Sentiment-Classifier/assets/123567363/e895c4c7-f8f4-4edf-97b1-0c3a12e8cefa)


從 Tuning Parameter 的過程中可以看到，隨著迭代次數的上升，Cost 和 Time elapsed 之間存在著取捨的關係，且所需時間隨著迭代次數呈指數上升。Cost 來到最低的 0.1013 的有兩組，分別是 `alpha = 1e-09, iter = 10000, time = 5.81 sec` , `alpha = 1e-11, iter = 1000000, time = 682.91 sec`，最後在時間成本的考量下我在後面的 Model Training 中選擇了 `alpha = 1e-09, iter = 10000` 這組參數。可以特別注意的是在 `alpha = 1e-12` 這組中，在 `iter < 50000` 的情況下 Cost 都還非常的大，原因是因為 Learning Rate 實在是太小了，導致 Gradient Descent 的速率太慢，一直到 `iter = 1000000` Cost 才下降到平均水準。

<h3> Model Training </h3>

```python
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, int(1e4))

t = []
for i in np.squeeze(theta):
    t.append(i)

print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {t}")
```

`The cost after training is 0.10133100`

`The resulting vector of weights is [3e-07, 0.00127474, -0.0011083]`

<h3> Validation </h3>

```python
def predict_tweet(tweet, freqs, theta):

    x = extract_features(tweet, freqs)

    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred
```

**Generate the sentences with the help of ChatGPT lol**

```python
vali_tweet = [

    "Another day, another opportunity.",

    "Do the right things, do things right.",

    "Celebrate the journey, not just the destination.",

    "Every sunset is an opportunity to reset.",

    "Stars can't shine without darkness.",

    "Inhale courage, exhale fear.",

    "Radiate kindness like sunshine.",

    "Find beauty in the ordinary.",

    "Chase your wildest dreams with the heart of a lion.",

    "Life is a canvas; make it a masterpiece.",

    "Let your soul sparkle.",

    "Create your own sunshine.", 

    "This summer would not be perfect without you." ]


for tweet in vali_tweet:
    print(process_tweet(tweet))
    print('%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))
    print('\n')
```

```python
Stem: ['anoth', 'day', 'anoth', 'opportun']
Another day, another opportunity. -> 0.533046


Stem: ['right', 'thing', 'thing', 'right']
Do the right things, do things right. -> 0.508265


Stem: ['celebr', 'journey', 'destin']
Celebrate the journey, not just the destination. -> 0.500568


Stem: ['everi', 'sunset', 'opportun', 'reset']
Every sunset is an opportunity to reset. -> 0.509145


Stem: ['star', 'shine', 'without', 'dark']
Stars can not shine without darkness. -> 0.499932


Stem: ['inhal', 'courag', 'exhal', 'fear']
Inhale courage, exhale fear. -> 0.500083


Stem: ['radiat', 'kind', 'like', 'sunshin']
Radiate kindness like sunshine. -> 0.515079


Stem: ['find', 'beauti', 'ordinari']
Find beauty in the ordinary. -> 0.506431


Stem: ['chase', 'wildest', 'dream', 'heart', 'lion']
Chase your wildest dreams with the heart of a lion. -> 0.496330


Stem: ['life', 'canva', 'make', 'masterpiec']
Life is a canvas; make it a masterpiece. -> 0.501446


Stem: ['let', 'soul', 'sparkl']
Let your soul sparkle. -> 0.514518


Stem: ['creat', 'sunshin']
Create your own sunshine. -> 0.502758


Stem: ['summer', 'would', 'perfect', 'without']
This summer would not be perfect without you. -> 0.509757
```

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

# Naive Bayes

<h3> Conditional Probability</h3>

不同於 Logistic Regression 是利用各項的權重去做判別 (等同於繪製一條類別分界線)，在 **Naive Bayes** 模型中，模型會逐字去計算該字為 Positive Sentiment 和 Negative Sentiment 的條件機率 $P(Pos | word)$ ，如 eq. 11 所示，又 $P('happy')$ 並不會因為 Pos / Neg 而改變，故最終 $P(Pos | word)$ 可以簡化為在 Positive Sentiment 下詞彙為 'happy' 的機率 $P('happy'|Positive)$ ，稱為 Likelihood，乘上這個詞彙本身是 Positive Sentiment 的機率 $P(Positive)$ ，稱為 Prior Assumption (eq. 12)。

$$
P(Positive\ |\ 'happy')=Probability\ of\ {\color{green}{positive}},\ given\ the\ word\ 'happy'\tag{11}
$$

$$
=\frac{P(Positive\ \cap\ 'happy')}{P('happy')}\ \ \ 
$$

$$
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ =P('happy'|Positive)*\frac{P(Positive)}{P('happy')}
$$

$$
P(Positive\ |\ 'happy') \propto {\color{blue}P('happy'|Positive)} * {\color{red}P(Positive)}\tag{12} 
$$

$$
{\color{blue}P('happy'|Positive)}:Likelihood\ \ \ \ \ {\color{red}P(Positive)}:Prior
$$

<h3> Naive Bayes Assumption </h3>

Naive Bayes Model 有兩個主要假設：

1. 詞彙所在的位置不重要，模型只會紀錄詞彙的性質 (即條件機率)，而不會紀錄其在 document 中的位置
2. 各詞彙之間的條件機率是獨立的，表示為 eq. 13

$$
P(f_1, f_2,......f_n | c) = P(f_1 | c) * P(f_2 | c) * ... * P(f_n | c) \tag{13}
$$

<h3> Naive Bayes Model </h3>

在開始做計算之前，Naive Bayes Model 會先將詞彙匯集成一袋的文字並統計各詞彙分別在 Positive / Negative Sentiment 句子中出現的次數，即可計算該詞彙的 Likelihood $P(w_i|class)$ ，為了避免有些詞彙只在單一 Sentiment 中出現會導致 $P(w_i|class) = 0$ 的現象，因此會進行 Laplacian Smoothing (eq. 14) 修正。修正過後的即可計算該詞彙的 Sentiment Ratio，將整句話的 Sentiment Ratio 相乘後就會得到模型對於這句話 Sentiment 的預測 (eq. 15)，但注意此時的 Sentiment Ratio $\in [0, \infty)$ ，如下圖會有不均衡量化 Negative Sentiment 的現象，若將 Likelihood 和 Prior 取 Log 後，會得到 Log Prior & Log Likelihood $\in (-\infty, \infty)$ ，即可更好的量化標準。要注意的是原本的 Sentiment Ratio 相乘 ($\prod$) 在取 Log 後要改成相加 ($\sum$)

$$
\tag{14} P(w_{class}) = \frac{freq_{class} + 1}{N_{class} + V}
$$

$$
freq_{class}: 特定詞彙在某類別中出現的詞頻\ (Ex: ('am', pos) = 3)
$$

$$
N_{class}: 某類別中的詞彙總數\ (N_{pos} = 13,\ N_{neg}=12)
$$

$$
V: Unique\ words\ in\ whole\ document\ (V=8)
$$

<img width="700" alt="截圖 2023-09-12 20 24 43" src="https://github.com/scfengv/NLP-Sentiment-Classifier/assets/123567363/f371e30c-6772-4a05-bed5-82abddb74376">

$$
\tag{15} Sentiment =  \frac{P(pos)}{P(neg)} * \frac{P(w_i|pos)}{P(w_i|neg)}
$$

$$
\tag{16} Sentiment =  \log(\frac{P(pos)}{P(neg)} * \prod_{i=1}^n \frac{P(w_i|pos)}{P(w_i|neg)}) = \log(\frac{P(pos)}{P(neg)}) + \sum_{i=1}^n \log(\frac{P(w_i|pos)}{P(w_i|neg)})
$$

$$
Sentiment =
\begin{cases}
    \begin{matrix}
        [0, \infty),\ Positive\\
        0,\ Neutral\\
        (-\infty, 0],\ Negative
    \end{matrix}
\end{cases}\tag{5}
$$

<img width="1500" alt="截圖 2023-09-12 20 24 43" src="https://github.com/scfengv/NLP-Sentiment-Classifier/assets/123567363/cfc07adf-8afa-4144-a9de-739e29e5345c">

總結以上，Naive Bayes Model 大致可以分成 4 個步驟，分別是

1. 文本前處理 (Cleaning, Word frequency count) (Rewind the **Data Preprocessing** part) 
2. 資料處理
    - Conditional Probability
    - Laplacian Smoothing
    - Log Likelihood & Log Prior
3. 計算並預測 Sentiment

**資料處理**

```python
def train_naive_bayes(freqs, train_x, train_y):
    
    data = {'word': [], 'positive': [], 'negative': [], 'sentiment': []}
    loglikelihood = {}
    logprior = 0

    # Calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0

    # Calculate Prior
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]

    D = len(train_y)
    D_pos = len(list(filter(lambda x: x > 0, train_y)))
    D_neg = len(list(filter(lambda x: x <= 0, train_y)))

    logprior = np.log(D_pos) - np.log(D_neg)

    # Calculate Likelihood
    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # Laplacian Smoothing
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

        if p_w_pos > p_w_neg:
            sentiment = 1
        else:
            sentiment = 0

        data['word'].append(word)
        data['positive'].append(np.log(p_w_pos))
        data['negative'].append(np.log(p_w_neg))
        data['sentiment'].append(sentiment)

    return logprior, loglikelihood, data
```

```python
LogPrior: 0.0  # Stands for pos/neg Balanced Dataset
Likelihood:
{'easili': -0.452940736126882,
 'melodi': 0.6456715525412289,
 'ohstylesss': 0.6456715525412289,
 'steelseri': -0.7406228085786619,
 'harsh': -0.7406228085786619,
 'weapon': -0.452940736126882,
 'maxdjur': -0.7406228085786619,
 'thalaivar': 0.6456715525412289,
 'theroyalfactor': 0.6456715525412289,
 'fought': 0.6456715525412289,
 'louisemensch': -0.7406228085786619,
 'hayli': 0.6456715525412289}
```

**計算並預測 Sentiment**

**Validation with ChatGPT tweet again**

```python
def naive_bayes_predict(tweet, logprior, loglikelihood):

    word_l = process_tweet(tweet)

    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:

            p += loglikelihood[word]

    return p
```

```python
Tweets: Another day, another opportunity.
Stem: ['anoth', 'day', 'anoth', 'opportun']
Another day, another opportunity. -> 2.267723


Tweets: Do the right things, do things right.
Stem: ['right', 'thing', 'thing', 'right']
Do the right things, do things right. -> -0.122857


Tweets: Celebrate the journey, not just the destination.
Stem: ['celebr', 'journey', 'destin']
Celebrate the journey, not just the destination. -> -0.324748


Tweets: Every sunset is an opportunity to reset.
Stem: ['everi', 'sunset', 'opportun', 'reset']
Every sunset is an opportunity to reset. -> 2.054798


Tweets: Stars can not shine without darkness.
Stem: ['star', 'shine', 'without', 'dark']
Stars can not shine without darkness. -> 0.572238


Tweets: Inhale courage, exhale fear.
Stem: ['inhal', 'courag', 'exhal', 'fear']
Inhale courage, exhale fear. -> -0.142427


Tweets: Radiate kindness like sunshine.
Stem: ['radiat', 'kind', 'like', 'sunshin']
Radiate kindness like sunshine. -> 1.410585


Tweets: Find beauty in the ordinary.
Stem: ['find', 'beauti', 'ordinari']
Find beauty in the ordinary. -> 1.288319


Tweets: Chase your wildest dreams with the heart of a lion.
Stem: ['chase', 'wildest', 'dream', 'heart', 'lion']
Chase your wildest dreams with the heart of a lion. -> -1.379487


Tweets: Life is a canvas; make it a masterpiece.
Stem: ['life', 'canva', 'make', 'masterpiec']
Life is a canvas; make it a masterpiece. -> 0.917726


Tweets: Let your soul sparkle.
Stem: ['let', 'soul', 'sparkl']
Let your soul sparkle. -> 1.488666


Tweets: Create your own sunshine.
Stem: ['creat', 'sunshin']
Create your own sunshine. -> 1.445494


Tweets: This summer would not be perfect without you.
Stem: ['summer', 'would', 'perfect', 'without']
This summer would not be perfect without you. -> 1.041158
```

`Naive Bayes model's accuracy = 0.9950`



### Reference
[1] [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/). Dan Jurafsky and James H. Martin Jan 7, 2023

[2] [Natural Language Processing with Classification and Vector Spaces](https://www.coursera.org/learn/classification-vector-spaces-in-nlp?specialization=natural-language-processing) DeepLearning.AI

[3] [Decoding Generative and Discriminative Models](https://www.analyticsvidhya.com/blog/2021/07/deep-understanding-of-discriminative-and-generative-models-in-machine-learning/) Chirag Goyal — Published On July 19, 2021 and Last Modified On September 13th, 2023

[4] [Gradient](https://en.wikipedia.org/wiki/Gradient) at Wikipedia
