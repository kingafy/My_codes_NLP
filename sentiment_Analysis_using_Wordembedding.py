# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:43:40 2020

@author: anshu
"""

##word embeddings for sentiment analysis
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, Flatten
from keras.datasets import imdb

num_words = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

print(len(X_train), 'train_data')
print(len(X_test), 'test_data')

max_len = 256
embedding_size = 32
batch_size = 128

print(X_train[0])
len(X_train[0])

pad  =  'post' #'pre'
#Convert our lists to equal length sequences
X_train_pad = pad_sequences(X_train, maxlen=max_len, padding=pad, truncating=pad)
X_test_pad = pad_sequences(X_test, maxlen=max_len, padding=pad, truncating=pad)

X_train_pad.shape

X_train_pad[0]

model = Sequential()
model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_len,
                    name='layer_embedding'))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.5))
# Final classification with a sigmoid:
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train_pad, y_train, epochs=5, validation_data=(X_test_pad, y_test), batch_size=batch_size)
eval_ = model.evaluate(X_test_pad, y_test)
print("Loss: {0:.5}".format(eval_[0]))
print("Accuracy: {0:.2%}".format(eval_[1]))