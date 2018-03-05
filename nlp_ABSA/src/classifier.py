# -*- coding: utf-8 -*-
"""
Created on Wed Feb  21 12:33:03 2018

@author: Yuan
"""

from src.cleaner import Cleaner
from src.preprocessingLSTM import proLSTM
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk import pos_tag, word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import optimizers
import pickle

#path for the pretrained vectors
GLOVE_DATASET_PATH = '../data/glove.twitter.27B.50d.txt'
#the dimension for word vectors, code of the aspect,
# code of the distance to the target word, code of POSTag
WORDVECTOR_DIM=50
ASPECT_DIM=12
DIST_TARGET=7
POS_DIM=36
DIM_WORD=WORDVECTOR_DIM+ASPECT_DIM+DIST_TARGET+POS_DIM
#dict for coding the aspect, POSTag
ASPECT={'AMBIENCE#GENERAL':0,'DRINKS#PRICES':1,'DRINKS#QUALITY':2,
        'DRINKS#STYLE_OPTIONS':3,'FOOD#PRICES':4,'FOOD#QUALITY':5,
        'FOOD#STYLE_OPTIONS':6,'LOCATION#GENERAL':7,'RESTAURANT#GENERAL':8,
        'RESTAURANT#MISCELLANEOUS':9,'RESTAURANT#PRICES':10,'SERVICE#GENERAL':11}
POSDICT={'CC':0,'CD':1,'DT':2,'EX':3,'FW':4,'IN':5,'JJ':6,'JJR':7,'JJS':8,'LS':9,'MD':10,'NN':11,'NNS':12,
         'NNP': 13,'NNPS':14,'PDT':15,'POS':16,'PRP':17,'PRP$':18,'RB':19,'RBR':20,'RBS':21,'RP':22,'SYM':23,'TO':24,'UH':25,
         'VB': 26,'VBD':27,'VBG':28,'VBN':29,'VBP':30,'VBZ':31,'WDT':32,'WP':33,'WP$':34,'WRB':35}


class Classifier:
    """The Classifier"""
    def __init__(self):
        #initialize the tokenizer
        self.tokenizer = Tokenizer(filters='')
        # set the default tag, used for replacing the words of dev data
        # which do not occur in train data
        self.defauttag='<hashtag>'
        # set an integer denoting the maximum length for each message,
        # for standardization in pad sequences
        self.maxlen_sent=100

    def clean(self,tfile):
        """
        clean data
        Parameters
        ----------
        tfile: string
            the path of the data needing to be processed

        Returns
        -------
        new_data: pd.DataFrame
        """

        # load data
        data = pd.read_csv(tfile, sep='\t', header=None,
                           names=['polarity', 'aspect', 'target', 'startend', 'message'])
        # clean the data
        cleaner = Cleaner()
        new_data = cleaner.remove_punctuation_dataframe(data)
        new_data = cleaner.remove_digits_dataframe(new_data)
        new_data = cleaner.lemmatization_dataframe(new_data)
        new_data = cleaner.lower_case(new_data)
        return new_data

    def inputlstm_line(self,df_line):
        """
        get the input matrix for one message of the targeted dataframe

        Parameters
        ----------
        df_line: pd.DtaFrame
            the data needing to be processed

        Returns
        -------
        tpmatrix.tolist(): list
            tpmatrix is the input matrix of this message for LSTM,
            whose shape= (len(cleandata),100,(50+12+5+36)),
            that is, whose shape= (self.maxlen_sent,DIM_WORD )
        """

        # generate POS Tag for one message
        POSlst=pos_tag(word_tokenize(df_line['message']))
        # pad sequences for this message
        seq_line=pad_sequences(self.tokenizer.texts_to_sequences([df_line['message']]), maxlen=self.maxlen_sent)
        tpmatrix=np.zeros((self.maxlen_sent,DIM_WORD))

        ##### add champ: aspect code
        aspectindex=ASPECT[df_line['aspect']]
        tpmatrix[:,WORDVECTOR_DIM+aspectindex]=1

        ##### add champ: distance according to target
        targets=df_line['target'].split()
        index_dict=self.tokenizer.word_index
        indexlst=[index_dict.get(word,None) for word in targets]
        # index in the tpmatrix
        tplst=[]
        for i in indexlst:
            tplst+=np.where(seq_line==i)[1].tolist()
        midpos=int(DIST_TARGET/2)
        tpmatrix[tplst,WORDVECTOR_DIM+ASPECT_DIM+midpos]=1
        if tplst[0] - 1 >= 0:
            tpmatrix[tplst[0]-1, WORDVECTOR_DIM + ASPECT_DIM + midpos-1] = 0.5
        if tplst[0] - 2 >= 0:
            tpmatrix[tplst[0]-2, WORDVECTOR_DIM + ASPECT_DIM + midpos-2] = 0.2
        if tplst[0] - 3 >= 0:
            tpmatrix[tplst[0]-3, WORDVECTOR_DIM + ASPECT_DIM + midpos-3] = 0.1

        if tplst[-1] + 1 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 1, WORDVECTOR_DIM + ASPECT_DIM + midpos+1] = 0.5
        if tplst[-1] + 2 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 2, WORDVECTOR_DIM + ASPECT_DIM + midpos+2] = 0.2
        if tplst[-1] + 3 <= self.maxlen_sent-1:
            tpmatrix[tplst[-1] + 3, WORDVECTOR_DIM + ASPECT_DIM + midpos+3] = 0.1


        ##### add champ: word vectors and POStag
        for i in range(1,self.maxlen_sent+1):
            tp=seq_line[0,-i]#wordcode in the pad seq
            # add champ: word vectors of 50d
            tpmatrix[-i, 0:WORDVECTOR_DIM] = self.wordvec_lookup[tp-1]
            # add champ: POStag
            if tp>0:
                posindex=POSDICT[POSlst[-i][1]]
                tpmatrix[-i, WORDVECTOR_DIM + ASPECT_DIM + DIST_TARGET+posindex] = 1

        return tpmatrix.tolist()


    def genInput_LSTM(self,cleandata):
        """should return a matrix, whose shape=(len(cleandata), 100, (50+12+5+36))
        that is, a matrix whose shape= (len(cleandata),self.maxlen_sent,DIM_WORD )
        as the input for lstm"""

        input_matrix=cleandata.apply(self.inputlstm_line,axis=1)
        return input_matrix.as_matrix()


    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        processor=proLSTM()
        ##### clean data and construct wordvector lookup table
        traindata = self.clean(trainfile)
        #add <hashtag> to the tokenizer
        train_addtag=traindata.copy()
        train_addtag['message'][0]+=' '+self.defauttag
        self.tokenizer.fit_on_texts(train_addtag['message'])
        #construct word vector look up table
        self.wordvec_lookup=processor.lookup_pretrained(GLOVE_DATASET_PATH,self.tokenizer,wordvector_dim=WORDVECTOR_DIM)
        #generate input for LSTM
        input_train = (self.genInput_LSTM(traindata)).tolist()

        train_labels = np.array(processor.getlabel(traindata).tolist())


        ##### lstm model
        model = Sequential()
        model.add(LSTM(DIM_WORD, input_shape=(self.maxlen_sent, DIM_WORD),dropout_W=0.3,dropout_U=0.3))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.99, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=rmsprop,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.history = model.fit(input_train, train_labels, epochs=25,verbose=0)
        #save the lstm model
        self.model=model

    def predict(self, devfile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        processor = proLSTM()
        ##### clean data and generate input for the lstm model
        devdata = self.clean(devfile)
        devdata=devdata.apply(processor.replace_wtag, axis=1,tokenizer=self.tokenizer,defauttag=self.defauttag)
        input_dev = (self.genInput_LSTM(devdata)).tolist()


        ##### predict
        pred_test=self.model.predict(input_dev)
        returnlst=np.apply_along_axis(processor.label_str, 1, pred_test)
        returnlst=returnlst.tolist()

        return returnlst





