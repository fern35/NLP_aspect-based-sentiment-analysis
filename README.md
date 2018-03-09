# NLP_aspect-based-sentiment-analysis
This is a project of the course NLP of Centralesupélec
reference link: http://nlpcourse.europe.naverlabs.com/#exercise2

# Description of the project
0) Introduction 
The goal of the project is to design a classifier to predict aspect-based polarities of opinions in
sentences, that assigns a polarity label to every triple <aspect categories, aspect_term, sentence>.
The polarity labels are positive, negative and neutral. Note that a sentence may have several
opinions.

For example, the input of the classifier is "AMBIENCE#GENERAL seating 18:25 short and sweet – seating is great:it's romantic,cozy and private." and the output of the classifier is "positive".

1) Data set
Each line contains 5 tab-separated fields: the polarity of the opinion, the aspect category on which
the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the
sentence in which that opinion is expressed.

an example:
"positive AMBIENCE#GENERAL seating 18:25 short and sweet – seating is great:it's romantic,cozy and private."

in this example, the opinion about the AMBIENCE#GENERAL aspect, which is associated to the term "seating", is positive.
There are 12 different aspects categories.

# Description of the code
## 0)pretrained word vectors
Download pretrained word vectors from: https://nlp.stanford.edu/projects/glove/

Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)

Choose the dimension: 50d 

## 1)Clean data
Before constructing input matrix for LSTM, I remove punctuation, digits from the data, and do lemmatization and transform all words to lower case. I do not remove stop words from the data, because I think it will break the structure of a complete sentence, which is probably not helpful for POSTags and training process of LSTM

There is a problem that there are some words in the dev set which do not occur in the train set. So I replace them with the string '\<hashtag\>'

#### There are three functions: lower_case(), remove_punctuation() and remove_digits(), which are based on the code of another project of Aeiocha and mine.

## 2) Construction of the input to the LSTM Model
I will explain the process with an example:
"negative SERVICE#GENERAL waitress 20:28 The hostess and the waitress were incredibly rude and did everything they could to rush us out."

After cleaning the data, we get "the hostess and the waitress were incredibly rude and did everything they could to rush us out" , and we get the aspect "SERVICE#GENERAL", aspect term "waitress".

For this message, we construct a 2D matrix: each row of the matrix represents the information of each word in the meassage. The last word in the meassage is "out", so the last row of the 2D matrix should be:
[the 50d word vector which represents "out", aspect code which represents "SERVICE#GENERAL", the distance code which represents the distance between "out" and aspect term "waitress", POSTag code which represents the POSTag of "out" in this message]

There is a problem that the number of words in each sentence is different, so we use the function "keras.preprocessing.sequence.pad_sequences()" to standardize the shape of the matrix. So the shape of the 2D matrix should be (100,(50+12+7+36))

Here, 100 is the maximum length of words that I choose to represent a message;  
50 is the dimension of pretrained word vector that we use;  
12 is the dimension of the vector for representing which aspect that we should focus on this message;  
7 is the dimension of the vector for representing the distance of this word to the target word;  
36 is the dimension of the vector for representing the POSTag for this word;  

## 2) model
I use LSTM model for training
The input of LSTM for training is a matrix whose shape=(1503,100,(50+12+7+36)), that means for each message, we construct a matrix whose shape=(100,(50+12+7+36)). 1503 is the number of messages of the train set.


The output of LSTM model is a vector with length=3;  
[1,0,0]-->'positive'  
[0,1,0]-->'negative'  
[0,0,1]-->'neutral'  

## 3). The accuracy  on the test set
The accuracy that I get on the dev set is 81 per cent

Please notice that the data is imbalanced, 70% of the message are "positive".

