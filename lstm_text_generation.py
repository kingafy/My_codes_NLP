# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:50:46 2020

@author: anshu
"""
import os
os.chdir(r"D:\Data science\NLP_learnings\jose portella\UPDATED_NLP_COURSE\06-Deep-Learning")
##LSTM text generation
def read_file(filepath):
    
    with open(filepath) as f:
        str_text = f.read()
    
    return str_text

read_file('moby_dick_four_chapters.txt')

import spacy
nlp = spacy.load('en',disable=['parser', 'tagger','ner'])

nlp.max_length = 1198623    

def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

d = read_file('melville-moby_dick.txt')
tokens = separate_punc(d)

len(tokens)

print(tokens)
# organize into sequences of tokens
train_len = 25+1 # 50 training words , then one target word

# Empty list of sequences
text_sequences = []

for i in range(train_len, len(tokens)):
    
    # Grab train_len# amount of characters
    seq = tokens[i-train_len:i]
    
    # Add to list of sequences
    text_sequences.append(seq)
    
print(text_sequences)

' '.join(text_sequences[0])
' '.join(text_sequences[1])


from keras.preprocessing.text import Tokenizer

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
sequences[0]

tokenizer.index_word

for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')
    
vocabulary_size = len(tokenizer.word_counts)
print(vocabulary_size)

import numpy as np
sequences = np.array(sequences)
sequences

import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model

from keras.utils import to_categorical

print(type(sequences))

# First 25 words
sequences[:,:-1]
sequences.shape

# last Word
sequences[:,-1]

X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocabulary_size+1)
seq_len = X.shape[1]
print(seq_len)

# define model
model = create_model(vocabulary_size+1, seq_len)
from pickle import dump,load

# fit model
model.fit(X, y, batch_size=128, epochs=300,verbose=1)
# save the model to file
model.save('epochBIG.h5')
# save the tokenizer
dump(tokenizer, open('epochBIG', 'wb'))