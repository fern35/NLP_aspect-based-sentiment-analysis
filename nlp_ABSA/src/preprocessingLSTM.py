# -*- coding: utf-8 -*-
"""
Created on Wed Feb  21 18:49:50 2018

@author: Yuan
"""
import numpy as np


class proLSTM(object):

    def lookup_pretrained(self,vectorpath,tokenizer,wordvector_dim):
        """
        Construct word vector look up table using the pretrained word vector

        Parameters
        ----------
        vectorpath: string
            the path of the pretrained vector file
        tokenizer: keras.preprocessing.text.Tokenizer
            the tokenizer for generating the word dictionary (word --> number)
        wordvector_dim: int
            the maximum length for every message (for standardization)

        Returns
        -------
        lookup_table: np.array((len(wordlist),wordvector_dim))
            the word vector look up table based on the words of the train set
        """
        #get the word dictionary
        word_dict=tokenizer.word_index
        wordlist=list(word_dict.keys())
        #initialize the look up table
        lookup_table=np.zeros((len(wordlist),wordvector_dim))
        # open the pretrained word vector file
        f = open(vectorpath,encoding='UTF8')
        # find the intersection of the pretrained words and the words in the train set
        # and build look up table according to the word dictionary
        for line in f:
            values = line.split(' ')
            word = values[0]
            if word in wordlist:
                vectors = np.asarray(values[1:], dtype='float32')
                lookup_table[word_dict[word]-1,:] = vectors
        f.close()
        return lookup_table

    def label_line(self,df_line):
        """
        transform the polarity to a vector, used in function "getlabel()"
        """
        if df_line['polarity']=='positive':
            return [1, 0, 0]
        elif df_line['polarity']=='negative':
            return [0,1,0]
        else:
            return [0,0,1]

    def getlabel(self,cleandata):
        """
        transform the polarity to a vector for every message of cleandata,
        which will be used for LSTM training
        """
        labels=cleandata.apply(self.label_line,axis=1)
        return labels

    def replace_wtag(self,df_line,tokenizer,defauttag):
        """
        replace the words which do not occur in the given word dictionary with the default tag,

        Parameters
        ----------
        df_line: pd.Series or pd.DataFrame
            the data needing to be processed
        tokenizer: keras.preprocessing.text.Tokenizer
            the tokenizer for generating the word dictionary (word --> number)
        defauttag: string
            the default tag which will replace the words that do not occur in the word dictionary

        Returns
        -------
        df_line: pd.Series or pd.DataFrame
        """

        message=df_line['message']
        message_lst=message.split(' ')
        targets=df_line['target']
        target_lst = targets.split(' ')
        for word in message_lst:
            if not word in tokenizer.word_index:
                message=message.replace(word,defauttag)
        for word in target_lst:
            if not word in tokenizer.word_index:
                targets=targets.replace(word,defauttag)

        df_line['message']=message
        df_line['target']=targets

        return df_line

    def label_str(self,labelvector):
        """
        transform the vector to polarity string
        """
        largestindex=np.argmax(labelvector)
        if largestindex==0:
            return 'positive'
        elif largestindex==1:
            return 'negative'
        else:
            return 'neutral'