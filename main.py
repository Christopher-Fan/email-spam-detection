import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
import random
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

def remove_punctuation(text):
    temp = str.maketrans('','', string.punctuation)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    word_list = []

    #check each word for if they're a stopword
    for word in str(text).split():
        word = word.lower()

        #if not a stop word, add it to our saved words
        if word not in stop_words:
            word_list.append(word)

    output = " ".join(word_list)
    
    return output

def main():
    #preprocessing
    data = pd.read_csv('spam_ham_dataset.csv')
    print(data.head())

    #show unbalanced data
    sns.countplot(x='label', data=data)
    plt.show()

    #splitting dataset into unflagged and flagged spam emails
    ham_msg = data[data['label'] == "ham"]
    spam_msg = data[data['label'] == "spam"]

    #balancing the dataset through random sampling
    ham_msg_sample = ham_msg.sample(n=len(spam_msg), random_state=random.randint(1,1000))
    balanced_data = pd.concat([ham_msg_sample, spam_msg]).reset_index(drop=True)

    #show rebalanced data
    sns.countplot(x='label', data=balanced_data)
    plt.title("Balanced Distribution of Spam and Normal Emails")
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Spam'])
    plt.show()

    #cleaning dataset
    balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
    balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuation(x))
    balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_stopwords(x))
    print(balanced_data.head())

if __name__ == "__main__":
    main()