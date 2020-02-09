# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:52:46 2020

@author: anshu
"""
import os
os.chdir(r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data")

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import string
import plotly.offline as py
import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.manifold import TSNE
#%matplotlib inline

df = pd.read_csv('yelp.csv')
df.head()
print(df['text'].head())

df= df.dropna()
df=df[['text','stars']]
df.head()

labels = df['stars'].map(lambda x : 1 if int(x) > 3 else 0)
print(labels[10:20])

def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text

df['text'] = df['text'].map(lambda x: clean_text(x))
df.head(10)

len(df)

maxlen = 50
embed_dim = 100
max_words = 20000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=maxlen, padding='post')
data[0]
vocab_size = len(tokenizer.word_index) + 1
vocab_size
labels = np.asarray(labels)

print('Shape of data:', data.shape)
print('Shape of label:', labels.shape)

##creating datasets
validation_split = .2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
val_samples = int(validation_split * data.shape[0])
X_train = data[:-val_samples]
y_train = labels[:-val_samples]
x_val = data[-val_samples:]
y_val = labels[-val_samples:]


dir = r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data\GloVe\glove.6B"
embed_index = dict()
f = open(os.path.join(dir, 'glove.6B.100d.txt'), encoding="utf8")


print(type(f))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embed_index[word] = coefs
f.close()


print('%s Word vectors' % len(embed_index))


##weight matrix creation
embed_matrix = np.zeros((max_words, embed_dim))

for word, i in tokenizer.word_index.items():
    if i < max_words:
        embed_vector = embed_index.get(word)
        if embed_vector is not None:
            embed_matrix[i] = embed_vector
            
##create model
model = Sequential()
model.add(Embedding(max_words,
                    embed_dim,
                    weights=[embed_matrix],
                    input_length=maxlen))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
save_best = ModelCheckpoint(r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data\models\yelp_comments.hdf", save_best_only=True, 
                               monitor='val_loss', mode='min')

model.fit(X_train, y_train,
          epochs=20,
          validation_data=(x_val, y_val),
          batch_size=128,
          verbose=1,
          callbacks=[early_stopping, save_best])

model.load_weights(filepath = r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data\models\yelp_comments.hdf")

pred = model.predict(x_val)

print(pred)