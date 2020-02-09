# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:38:12 2020

@author: anshu
"""

##LSTM/GRU on sentiment analysis
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, CuDNNLSTM, CuDNNGRU, Dropout
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

num_words = 20000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

print(len(X_train), 'train_data')
print(len(X_test), 'test_data')

print(X_train[0])

len(X_train[0])

print(y_train)
max_len = 256
embedding_size = 10
batch_size = 128
n_epochs = 10


pad  =  'pre' #'post'

X_train_pad = pad_sequences(X_train, maxlen=max_len, padding=pad, truncating=pad)
X_test_pad = pad_sequences(X_test, maxlen=max_len, padding=pad, truncating=pad)

X_train_pad[0]

model = Sequential()

#The input is a 2D tensor: (samples, sequence_length)
# this layer will return 3D tensor: (samples, sequence_length, embedding_dim)
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_len,
                    name='layer_embedding'))

model.add(Dropout(0.2))

#model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))

model.add(CuDNNLSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', name='classification'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train_pad, y_train,
          epochs=n_epochs,
          batch_size=batch_size, 
          validation_split=0.05,
          callbacks=[callback_early_stopping]
         )

eval_ = model.evaluate(X_test_pad, y_test)

print("Loss: {0:.5}".format(eval_[0]))
print("Accuracy: {0:.2%}".format(eval_[1]))