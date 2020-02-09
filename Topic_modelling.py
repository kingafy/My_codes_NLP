# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:11:55 2020

@author: anshu
"""

import pandas as pd
import os
os.chdir(r"D:\Data science\NLP_learnings\jose portella\UPDATED_NLP_COURSE\05-Topic-Modeling")
npr = pd.read_csv('npr.csv')
npr.head()

len(npr)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = cv.fit_transform(npr['Article'])

dtm

from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)

# This can take awhile, we're dealing with a large amount of documents!
LDA.fit(dtm)

print(cv.get_feature_names())
len(cv.get_feature_names())

len(LDA.components_)

LDA.components_

len(LDA.components_[0])

single_topic = LDA.components_[0]
# Returns the indices that would sort this array.
single_topic.argsort()
# Word least representative of this topic
single_topic[18302]

# Word most representative of this topic
single_topic[42993]

# Top 10 words for this topic:
single_topic.argsort()[-10:]
top_word_indices = single_topic.argsort()[-10:]

for index in top_word_indices:
    print(cv.get_feature_names()[index])
    
##let's view all the  topics found.
    
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
####################################33
#Attaching Discovered Topic Labels to Original Articles

topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0]
topic_results[0].round(2)
index of topic which has the max probabilty
topic_results[0].argmax()

npr.head()
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)

npr.head()

###############NON NEGATIVE MATRIX FACTORIZATION######
os.chdir(r"D:\Data science\NLP_learnings\jose portella\UPDATED_NLP_COURSE\05-Topic-Modeling")
npr = pd.read_csv('npr.csv')
npr.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])
dtm
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7,random_state=42)
# This can take awhile, we're dealing with a large amount of documents!
nmf_model.fit(dtm)
len(tfidf.get_feature_names())
import random
len(nmf_model.components_)
nmf_model.components_
len(nmf_model.components_[0])
single_topic = nmf_model.components_[0]
# Returns the indices that would sort this array.
single_topic.argsort()
# Word least representative of this topic
single_topic[18302]

# Word most representative of this topic
single_topic[42993]

# Top 10 words for this topic:
single_topic.argsort()[-10:]

top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(tfidf.get_feature_names()[index])
    
for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
topic_results = nmf_model.transform(dtm)

topic_results.shape

topic_results[0]
topic_results[0].round(2)
topic_results[0].argmax()

topic_results.argmax(axis=1)

npr['Topic'] = topic_results.argmax(axis=1)
npr.head(10)