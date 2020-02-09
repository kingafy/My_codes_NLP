# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:21:07 2020

@author: anshu
"""

##fast text with Gensim
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
import matplotlib.pyplot as plt
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.corpus import gutenberg

#nltk.download('gutenberg')
gutenberg.fileids()

#bible = gutenberg.sents('bible-kjv.txt')
bible = gutenberg.raw('bible-kjv.txt')
bible

bible_sents = sent_tokenize(bible)
bible_sents

print('Total lines:', len(bible_sents))

##cleaning text
remove_terms = punctuation + '0123456789'

def preprocessing(text):
    words = word_tokenize(text)
    tokens = [w for w in words if w.lower() not in remove_terms]
    stopw = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopw]
    # remove words less than three letters
    tokens = [word for word in tokens if len(word)>=3]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # lemmatize
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(word) for word in tokens]    
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

corpus = [preprocessing(sentence) for sentence in bible_sents if sentence.strip() !='']
corpus

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = [wpt.tokenize(doc) for doc in corpus]
tokenized_corpus[1]

feature_size = 50   # Word embedding vector dimensionality  
window_context = 30  # Context window size                                                                                    
min_word_count = 5   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words
fasttext_model = FastText(tokenized_corpus,
                          size=feature_size,
                          window=window_context,
                          min_count=min_word_count,
                          sample=sample,
                          sg=1, # sg decides whether to use the skip-gram model (1) or CBOW (0)
                          iter=50)

print(fasttext_model.wv['god'])

similar_words = {search_term: [item[0] for item in fasttext_model.wv.most_similar([search_term], topn=5)]
                  for search_term in ['god', 'jesus', 'noah', 'egypt', 'john', 'gospel', 'moses']}
similar_words 

print(fasttext_model.wv.similarity(w1='god', w2='satan'))
sentence_1 = "satan god jesus peter"
print('Don´t belongs to [',sentence_1, ']:',  
      fasttext_model.wv.doesnt_match(sentence_1.split()))

sentence_2 = "john james judas jesus"
print('Don´t belongs to [',sentence_2, ']:', 
      fasttext_model.wv.doesnt_match(sentence_2.split()))

fasttext_model.save(r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data\models\saved_fasttext_model_gensim")

loaded_model = FastText.load(r"D:\Data science\NLP_learnings\deep learning NLP\DEEP_NLP_resources\data\models\saved_fasttext_model_gensim")
print(loaded_model)