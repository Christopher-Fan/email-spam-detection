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

def plot_word_cloud(data, classifier):
    email_corpus = " ".join(data['text'])
    wc = WordCloud(background_color='black', max_words=100, width=800, height=400).generate(email_corpus)
    plt.figure(figsize=(7,7))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {classifier} Emails', fontsize=15)
    plt.axis('off')
    plt.savefig(f'word_cloud_{classifier}.png')
    plt.show()

def main():
    #preprocessing
    data = pd.read_csv('spam_ham_dataset.csv')
    print(data.head())

    #show unbalanced data
    sns.countplot(x='label', data=data)
    plt.savefig('unbalanced_data.png')
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
    plt.savefig('rebalanced_data.png')
    plt.show()

    #cleaning dataset
    balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
    balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuation(x))
    balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_stopwords(x))
    print(balanced_data.head())

    #show flagged email wordcloud to highlight differences in word corpus
    plot_word_cloud(balanced_data[balanced_data['label'] == 'ham'], classifier='Non-Spam')
    plot_word_cloud(balanced_data[balanced_data['label'] == 'spam'], classifier='Spam')

    #create train and test split
    train_x, test_x, train_y, test_y = train_test_split(
        balanced_data['text'],
        balanced_data['label'], 
        test_size=0.2,
        random_state=random.randint(1,1000)
    )

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)

    train_sequences = tokenizer.texts_to_sequences(train_x)
    test_sequences = tokenizer.texts_to_sequences(test_x)

    #Padding sequence sentences to 100 characters
    train_sequences = pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')

    train_y = (train_y == 'spam').astype(int)
    test_y = (test_y == 'spam').astype(int)

    #Sequential Model for output prediction of spam or not spam
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=100),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') #output layer
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )

    model.build(input_shape=(None, 100))
    model.summary()

    
    es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
    lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

    history = model.fit(
        train_sequences, train_y,
        validation_data=(test_sequences, test_y),
        epochs=20,
        batch_size=32,
        callbacks=[lr, es]
    )

    test_loss, test_accuracy = model.evaluate(test_sequences, test_y)
    print('Test Loss :',test_loss)
    print('Test Accuracy :',test_accuracy)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()