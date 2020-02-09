# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:24:51 2019

@author: anshu
"""

###nlp_tasks


##Text preprocessing
'''
Noise Removal
'''
noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

print(_remove_noise("this is a sample text"))

##Alternative approach using stopwords of nltk

from nltk.corpus import stopwords
noise_list = stopwords.words('english')
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

print(_remove_noise("this is a sample text"))

###regex based cleaning approach

import re 

def _remove_regex(input_text, regex_pattern):
    urls = re.finditer(regex_pattern, input_text) 
    for i in urls: 
        input_text = re.sub(i.group().strip(), '', input_text)
    return input_text

regex_pattern = "#[\w]*"  

print(_remove_regex("remove this #hashtag from analytics vidhya", regex_pattern))
                    
                    
###Normalization of Text --Lexicon normalization
'''
The most common lexicon normalization practices are :

Stemming:  Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.
Lemmatization: Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations).
'''
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")
 
stem.stem(word)

###dictionary based normalization process//synonyms definition based approach
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}
def _lookup_words(input_text):
    words = input_text.split() 
    new_words = [] 
    for word in words:
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
        new_words.append(word) 
    new_text = " ".join(new_words) 
    return new_text

print(_lookup_words("RT this is a retweeted tweet by Shivam Bansal"))


####html characters removal
'''
Escaping HTML characters: Data obtained from web usually contains a lot of html entities like &lt; &gt; &amp; which gets embedded in the original data. It is thus necessary to get rid of these entities. One approach is to directly remove them by the use of specific regular expressions. Another approach is to use appropriate packages and modules (for example htmlparser of Python), which can convert these entities to standard html tags. For example: &lt; is converted to “<” and &amp; is converted to “&”.
'''
original_tweet = "I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, sooo happppppy 🙂 http://www.apple.com"

print('\n\nEscaping HTML Characters\n\n')

from html.parser import HTMLParser
html_parser = HTMLParser()
tweet = html_parser.unescape(original_tweet)
print("original tweet",original_tweet)
print(tweet)



###decoding data
'''
 Thisis the process of transforming information from complex symbols to simple and easier to understand characters. Text data may be subject to different forms of decoding like “Latin”, “UTF8” etc. Therefore, for better analysis, it is necessary to keep the complete data in standard encoding format. UTF-8 encoding is widely accepted and is recommended to use.
 '''
 print(original_tweet)
 tweet = original_tweet.encode("ascii","ignore")
 print(tweet)
 
 '''
 Apostrophe Lookup: To avoid any word sense disambiguation in text, it is recommended to maintain proper structure in it and to abide by the rules of context free grammar. When apostrophes are used, chances of disambiguation increases.
For example “it’s is a contraction for it is or it has”.
'''
APPOSTOPHES = {"'s" : " is", "’re" : " are"} ## Need a huge dictionary
words = original_tweet.split()
print(words)
reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
print(reformed)
reformed = " ".join(reformed)
print(reformed)


###POS tagging
from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print(pos_tag(tokens))

###ngrams
def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(words[i:i+n])
    return output

generate_ngrams('this is a sample text', 2)