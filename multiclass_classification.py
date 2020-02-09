# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:58:53 2019

@author: anshu
"""

import numpy as np 
import pandas as pd
import re
import gc
import os
#print(os.listdir("D:\Data science\NLP_learnings\Bert"))
os.chdir(r"D:\Data science\NLP_learnings\Bert")
import fileinput
import string
import tensorflow as tf
import zipfile
import datetime
import sys
from tqdm  import tqdm
tqdm.pandas()
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score

from tensorflow.python.keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report

valid_forms = ['am','are','were','was','is','been','being','be']
blank = '----'
def detect(tokens):
    return [t for t in tokens if t in valid_forms]
    
def replace_blank(tokens):
    return [blank if t in valid_forms else t for t in tokens]

def create_windows(tokens, window_size=3):
    X = []
    for i, word in enumerate(tokens):
        if word == blank:
            window = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
            window = ' '.join(window)
            X.append(window)    
    return X

f1 = open("internet_archive_scifi_v3.txt","r")
a = f1.read()
a = re.sub('[\n]', '', a)
tokens = wordpunct_tokenize(a)
y = detect(tokens)
tokens = replace_blank(tokens)
X = create_windows(tokens)
f1.close()

df = pd.DataFrame()
df["Text"] = X
df["Label"] = y

print(df.head())

one_hot = pd.get_dummies(df["Label"])
df.drop(['Label'],axis=1,inplace=True)
df = pd.concat([df,one_hot],axis=1)
df.head()


df1 = df[:200000]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1["Text"].values, df1.drop(['Text'],axis=1).values, test_size=0.2, random_state=42)


vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=50)

print(X_train)


sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences, maxlen=50)


model = tf.keras.Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(8, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train,
                    batch_size=1024,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(X_test, y_test,
                       batch_size=256, verbose=1)
print('Test accuracy:', score[1])

preds = model.predict(X_test)

print(classification_report(np.argmax(y_test,axis=1),np.argmax(preds,axis=1)))