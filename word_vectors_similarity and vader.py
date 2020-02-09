# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:24:13 2020

@author: anshu
"""

import spacy
#package for vectors ---en_core_web_md
nlp = spacy.load('en_core_web_md')
nlp(u'lion').vector

doc = nlp(u'The quick brown fox jumped over the lazy dogs.')

doc.vector
len(doc.vector)

##identify similarity between vectors
# Create a three-token Doc object:
tokens = nlp(u'lion cat pet')

# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
##another approach of doing similarity
nlp(u'lion').similarity(nlp(u'dandelion'))

# =============================================================================
# Vector norms it's sometimes helpful to aggregate 300 dimensions into a Euclidian (L2) norm, computed as the square root of the sum-of-squared-vectors. This is accessible as the .vector_norm token attribute. Other helpful attributes include .has_vector and .is_oov or out of vocabulary.
# =============================================================================
tokens = nlp(u'dog cat nargle')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    
##vector arithmetic
    
from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])


##sentiment analysis using Vader sentiment
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

a = 'This was a good movie.'
sid.polarity_scores(a)

a = 'This was the best, most awesome movie EVER MADE!!!'
sid.polarity_scores(a)

a = 'This was the worst film to ever disgrace the screen.'
sid.polarity_scores(a)


import numpy as np
import pandas as pd
import os
os.chdir(r"D:\Data science\NLP_learnings\jose portella\UPDATED_NLP_COURSE\04-Semantics-and-Sentiment-Analysis")
df = pd.read_csv('amazonreviews.tsv', sep='\t')
df.head()

df['label'].value_counts()

##remove empty records 
# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

df.drop(blanks, inplace=True)


df['label'].value_counts()

sid.polarity_scores(df.loc[0]['review'])
df.loc[0]['label']

##adding scores and labels to dataframe
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

df.head()

##extract the compound score into a separate column



df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
df.head()

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

df.head()

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy_score(df['label'],df['comp_score'])

print(confusion_matrix(df['label'],df['comp_score']))
