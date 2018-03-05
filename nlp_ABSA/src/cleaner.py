# -*- coding: utf-8 -*-
"""
Created on Wed Feb  21 15:05:28 2018

@author: Yuan
"""
import string
from nltk.stem import WordNetLemmatizer


class Cleaner(object):
    def lower_case(self, data):
        """
        Lower the string of data['target'] and data['message']

        Parameters
        ----------
        data: pd.DataFrame

        Returns
        -------
        new_data: pd.DataFrame
        """
        new_data = data.copy()
        new_data['target'] = new_data['target'].apply(lambda x: x.lower())
        new_data['message'] = new_data['message'].apply(lambda x: x.lower())

        return new_data

    def remove_punctuation_dataframe(self, data):
        """
        Remove punctuation in data['target'] and data['message']

        Parameters
        ----------
        data: pd.DataFrame

        Returns
        -------
        new_data: pd.DataFrame
        """
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.remove_punctuation(x))
        new_data['target'] = new_data['target'].apply(
            lambda x: self.remove_punctuation(x))

        return new_data

    def remove_punctuation(self, data):
        """
        Remove punctuation inside the object, used
        in the function "remove_punctuation_dataframe()"

        Parameters
        ----------
        data: str or unicode

        Returns
        -------
        new_data: str or unicode
        """
        str_lang = string.punctuation
        for punctuation in str_lang:
            data = data.replace(punctuation, ' ')
        new_data = ' '.join(data.split())
        return new_data

    def remove_digits_dataframe(self, data):
        """
        Remove punctuation in data['target'] and data['message']

        Parameters
        ----------
        data: np.DataFrame

        Returns
        -------
        new_data: pd.DataFrame
        """
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.remove_digits(x))
        new_data['target'] = new_data['target'].apply(
            lambda x: self.remove_digits(x))

        return new_data

    def remove_digits(self, data):
        """
        Remove digits inside the object, used
        in the function "remove_digits_dataframe()"

        Parameters
        ----------
        data: str or unicode

        Returns
        -------
        new_data: str or unicode
        """
        str_lang = string.digits
        for punctuation in str_lang:
            data = data.replace(punctuation, ' ')
        new_data = ' '.join(data.split())
        return new_data

    def lemmatization_dataframe(self, data):
        """
        Lemmatize in data['target'] and data['message']

        Parameters
        ----------
        data: pd.DataFrame

        Returns
        -------
        new_data pd.DataFrame
        """
        new_data = data.copy()
        new_data['message'] = new_data['message'].apply(
            lambda x: self.lemmatization(x))

        new_data['target'] = new_data['target'].apply(
            lambda x: self.lemmatization(x))

        return new_data

    def lemmatization(self, data):
        """
        Method of lemmatization for one object, used in function "lemmatization_dataframe()"

        Parameters
        ----------
        data: str or unicode

        Returns
        -------
        new_data: str or unicode
        """
        wordnet_lemmatizer = WordNetLemmatizer()
        words = data.split()
        new_data = []

        for word in words:
            word = wordnet_lemmatizer.lemmatize(word, 'v')
            new_data.append(word)

        new_data = ' '.join(new_data)

        return new_data



