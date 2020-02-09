# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:54:24 2020

@author: anshu
"""

##skipgram_word2VEc
import os
os.chdir(r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data")
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input
from keras.layers.merge import Dot
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
import gensim
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
remove_terms = punctuation + '0123456789'

def preprocessing(text):
    words = word_tokenize(text)
    tokens = [w for w in words if w.lower() not in remove_terms]
    #stopw = stopwords.words('english')
    #tokens = [token for token in tokens if token not in stopw]
    # remove words less than three letters
    #tokens = [word for word in tokens if len(word)>=3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # lemmatize
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

vocab_size = len(tokenizer.word_index) + 1
vocab_size
items = tokenizer.word_index.items()

dim_embedddings = 300

# inputs
inputs = Input(shape=(1, ), dtype='int32')
w = Embedding(vocab_size, dim_embedddings)(inputs)

# context
c_inputs = Input(shape=(1, ), dtype='int32')
c  = Embedding(vocab_size, dim_embedddings)(c_inputs)

d = Dot(axes=2)([w, c])

d = Reshape((1,), input_shape=(1, 1))(d)
d = Activation('sigmoid')(d)

model = Model(inputs=[inputs, c_inputs], outputs=d)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
n_epochs = 15
for epoch in range(n_epochs):
    loss = 0.
    for i, doc in enumerate(X_train_tokens):
        data, labels = skipgrams(sequence=doc, vocabulary_size=vocab_size, window_size=4)
        x = [np.array(x) for x in zip(*data)]
        y = np.array(labels, dtype=np.int32)
        if x:
            loss += model.train_on_batch(x, y)

    print('Epoch:', epoch, '\tLoss:', loss)
    
f = open('word2vec-skipgrams1.txt' ,'w', encoding="utf8")
f.write('{} {}\n'.format(vocab_size-1, dim_embedddings))

weights = model.get_weights()[0]
for word, i in items:
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(weights[i, :])))))
f.close()


###loading the model
w2v = gensim.models.KeyedVectors.load_word2vec_format('word2vec-skipgrams1.txt', binary=False)
w2v.most_similar(positive=['solar'])
w2v.most_similar(positive=['system'])

w2v.most_similar(positive=['kepler'])