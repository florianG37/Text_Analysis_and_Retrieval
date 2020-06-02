import collections

import pandas

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

import numpy as np
import matplotlib.pyplot as plt

# First define a function to find all words (excluding numbers and stopwords) of each category
def getWords(label):
    temp_words = ' '.join(list(dataset.loc[dataset['label'] == label]['SMS'])) 
    lst_words = []
    words = [word.lower() for word in word_tokenize(temp_words) 
             if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    lst_words = lst_words + words
    return lst_words



dataset = pandas.read_csv("SMSSpamCollection.txt", sep="	", header=None)
dataset.columns = ["label", "SMS"]

# Get both spam and ham words
spam_words = getWords('spam')
ham_words = getWords('ham')


# 10 most frequent spam words
Counter = collections.Counter(spam_words)
most_occur_spam_words = Counter.most_common(10)
dataset_most_occur_spam_words = pandas.DataFrame(most_occur_spam_words, columns=['word','frequency'])
dataset_most_occur_spam_words.plot(x='word', y='frequency', kind='bar', figsize=(15, 7), color = 'red')



# 10 most frequent ham words
Counter = collections.Counter(ham_words)
most_occur_ham_words = Counter.most_common(10)
dataset_most_occur_ham_words = pandas.DataFrame(most_occur_ham_words, columns=['word','frequency'])
dataset_most_occur_ham_words.plot(x='word', y='frequency', kind='bar', figsize=(15, 7), color = 'blue')

plt.show()