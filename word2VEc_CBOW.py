# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:22:17 2020

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
from keras.layers import Embedding, Dense, Lambda
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras.backend as K
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer #takes into consideration the morphological analysis of the words
from nltk.stem.porter import PorterStemmer ##cutting off the end or the beginning 

remove_terms = punctuation + '0123456789'

def preprocessing(text):
    words = word_tokenize(text)
    tokens = [w for w in words if w.lower() not in remove_terms]
    #stopw = stopwords.words('english')
    #tokens = [token for token in tokens if token not in stopw]
    #tokens = [word for word in tokens if len(word) >= 3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]    
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

corpus = open("History_of_Astronomy.txt", encoding="utf8").readlines()
corpus = [preprocessing(sentence) for sentence in corpus if sentence.strip() !='']
corpus

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
X_train_tokens = tokenizer.texts_to_sequences(corpus)
items = tokenizer.word_index.items()
word2id = tokenizer.word_index
id2word = dict([(value, key) for (key, value) in word2id.items()])
vocab_size = len(word2id) + 1
vocab_size
embed_size = 300
window_size = 2

def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])   
            
            label_word.append(word)

            x = pad_sequences(context_words, maxlen=context_length)
            y = to_categorical(label_word, vocab_size)
            yield (x, y)
            
i = 0
for x, y in generate_context_word_pairs(corpus=X_train_tokens, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
    
        if i == 10:
            break
        i += 1
        
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embed_size,
                   embeddings_initializer='glorot_uniform',
                   input_length=window_size*2))
model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))

model.add(Dense(vocab_size, kernel_initializer='glorot_uniform', activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam')

n_epochs = 10
for epoch in range(n_epochs):
    loss = 0.
    for x, y in generate_context_word_pairs(corpus=X_train_tokens, window_size=window_size, vocab_size=vocab_size):
        loss += model.train_on_batch(x, y)

    print('Epoch:', epoch, '\tLoss:', loss)
    
weights = model.get_weights()[0]
weights = weights[1:]
print(weights.shape)
pd.DataFrame(weights, index=list(id2word.values())).head(30)

distance_matrix = cosine_similarity(weights)
print(distance_matrix.shape)

similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1] 
                   for search_term in ['copernicus', 'system', 'sun', 'halley', 'kepler','discovery','ancient']}
similar_words

