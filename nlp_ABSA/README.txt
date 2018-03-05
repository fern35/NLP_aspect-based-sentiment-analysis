1. Authors: Yuan ZHANG, Jacques Manderscheid, Louis Philippe
Emails: yuan.zhang@supelec.fr, jacques.manderscheid@supelec.fr, louis.philippe@supelec.fr

2. Description
0)pretrained word vectors
we download pretrained word vectors from: https://nlp.stanford.edu/projects/glove/

Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)

We choose the dimension: 50d 

1)Clean data
Before constructing input matrix for LSTM, we remove punctuation, digits from the data, and do lemmatization and transform all words to lower case. We do not remove stop words from the data, because we think it will break the structure of a complete sentence, which is probably not helpful for POSTags and training process of LSTM

There is a problem that there are some words in the dev set which do not occur in the train set. So we replace them with the string '<hashtag>'

2) model
We use LSTM model for training
The input of LSTM for training is a matrix whose shape=(1503,100,(50+12+7+36)), that means for each message, we construct a matrix whose shape=(100,(50+12+7+36)). 

Here, 100 is the maximum length of words that we choose to represent a message;
50 is the dimension of pretrained word vector that we use
12 is the dimension of the vector for representing which aspect that we should focus on this message;
7 is the dimension of the vector for representing the distance of this word to the target word
36 is the dimension of the vector for representing the POSTag for this word

The output of LSTM model is a vector with length=3;
[1,0,0]-->'positive'
[0,1,0]-->'negative'
[0,0,1]-->'neutral'

3. The accuracy that we get on the dev set is 81 per cent

