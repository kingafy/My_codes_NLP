# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:24:50 2019

@author: anshu
"""

#####NLTK hands on ######


from nltk.tokenize import word_tokenize
from nltk.text import Text

my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."

###tokenize string
tokens = word_tokenize(my_string)
print(tokens)

### convert tokens to lower case
tokens_lower = [token.lower() for token in tokens]

print(tokens_lower)


t = Text(tokens)
print(t)

###concordance
t.concordance('uckers') # concordance() is a method of the Text class of NLTK. It finds words and displays a context window. Word matching is not case-sensitive.
# concordance() is defined as follows: concordance(self, word, width=79, lines=25). Note default values for optional params.

t.collocations() # def collocations(self, num=20, window_size=2). num is the max no. of collocations to print.


####count nof times a word present in a string
t.count("quack")

t.index("two")


##find similar words in the context
t.similar('brother') # similar(self, word, num=20). Distributional similarity: find other words which appear in the same contexts as the specified word; list most similar words first.

# Reveals patterns in word positions. Each stripe represents an instance of a word, and each row represents the entire text.
t.dispersion_plot(['man', 'thing', 'quack']) 

t.plot(100)


###frequency distribution of words in the entire vocabulary
t.vocab()


from nltk.corpus import reuters
text = Text(reuters.words()) # .words() is one method corpus readers provide for reading data from a corpus. We will learn more about these methods in Chapter 2.
text.common_contexts(['August', 'June']) # It seems that .common_contexts() takes 2 words which are used similarly and displays where they are used similarly. It also seems that '_' indicates where the words would be in the text.


###Ngrams

s = "Le temps est un grand maître, dit-on, le malheur est qu'il tue ses élèves."
s = s.lower()
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")
s_tokenized = tokenizer.tokenize(s)
s_tokenized



####ngrams

s = "Le temps est un grand maître, dit-on, le malheur est qu'il tue ses élèves."
s = s.lower()
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")
s_tokenized = tokenizer.tokenize(s)
s_tokenized

from nltk.util import ngrams
generated_4grams = []

for word in s_tokenized:
    generated_4grams.append(list(ngrams(word, 4, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_'))) # n = 4.
generated_4grams


##obtaining ngrams

ng_list_4grams = generated_4grams
for idx, val in enumerate(generated_4grams):
    ng_list_4grams[idx] = ''.join(val)
ng_list_4grams


##sorting ngrams  by frequency (n=4)

freq_4grams = {}

for ngram in ng_list_4grams:
    if ngram not in freq_4grams:
        freq_4grams.update({ngram: 1})
    else:
        ngram_occurrences = freq_4grams[ngram]
        freq_4grams.update({ngram: ngram_occurrences + 1})
        
from operator import itemgetter # The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python. For example, operator.add(x, y) is equivalent to the expression x + y.

freq_4grams_sorted = sorted(freq_4grams.items(), key=itemgetter(1), reverse=True)[0:300] # We only keep the 300 most popular n-grams. This was suggested in the original paper written about n-grams.
freq_4grams_sorted




#####TOKENIZING######
text = "Yo man, it's time for you to shut yo' mouth! I ain't even messin' dawg."

import sys

try:
    from nltk.tokenize import wordpunct_tokenize # RE-based tokenizer which splits text on whitespace and punctuation (except for underscore)
except ImportError:
    print('[!] You need to install nltk (http://nltk.org/index.html)')
    

test_tokens = wordpunct_tokenize(text)
test_tokens


#### stopwords#######

from nltk.corpus import stopwords
stopwords.readme().replace('\n', ' ') # Since this is raw text, we need to replace \n's with spaces for it to be readable.

stopwords.fileids() # Most corpora consist of a set of files, each containing a piece of text. A list of identifiers for these files is accessed via fileids().


stopwords.words('english')[:10]

len(stopwords.words('english'))

###
##We loop through the list of stop words in all languages and check how many stop words our test text contains in each language. The text is then classified to be in the language in which it has the most stop words.
language_ratios = {}

test_words = [word.lower() for word in test_tokens] # lowercase all tokens
test_words_set = set(test_words)

for language in stopwords.fileids():
    stopwords_set = set(stopwords.words(language)) # For some languages eg. Russian, it would be a wise idea to tokenize the stop words by punctuation too.
    common_elements = test_words_set.intersection(stopwords_set)
    language_ratios[language] = len(common_elements) # language "score"
    
language_ratios


most_rated_language = max(language_ratios, key=language_ratios.get) # The key parameter to the max() function is a function that computes a key. In our case, we already have a key so we set key to languages_ratios.get which actually returns the key.
most_rated_language

test_words_set.intersection(set(stopwords.words(most_rated_language))) # We can see which English stop words were found.

######################fiding unusual words in given language
text = "Truly Kryptic is the best puzzle game. It's browser-based and free. Google it."
from nltk import word_tokenize
text_tokenized = word_tokenize(text.lower())
text_tokenized

###importing words
from nltk.corpus import words
words.readme().replace('\n', ' ')
words.fileids()

words.words('en')[:10]
words.words('en-basic')[:10]

##finding unusual words

english_vocab = set(w.lower() for w in words.words())
text_vocab = set(w.lower() for w in text_tokenized if w.isalpha()) # Note .isalpha() removes punctuation tokens. However, tokens with a hyphen like 'browser-based' are totally skipped over because .isalpha() would be false.
unusual = text_vocab.difference(english_vocab)
unusual


####creating POS tagger

##We can train a classifier to work out which suffixes are most informative for POS tagging. We can begin by finding out what the most common suffixes are

from nltk.corpus import brown
from nltk import FreqDist

suffix_fdist = FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1
    
suffix_fdist

common_suffixes = [suffix for (suffix,count) in suffix_fdist.most_common(100)]
common_suffixes[:10]


##define a feature extractor function which checks a given word for these suffixes:
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

pos_features('test')

##feature extractor, we can use it to train a new decision tree classifier:
tagged_words = brown.tagged_words(categories='news')

print(tagged_words)
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
featuresets[0]

from nltk import DecisionTreeClassifier
from nltk.classify import accuracy

cutoff = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[cutoff:], featuresets[:cutoff]

classifier = DecisionTreeClassifier.train(train_set) # NLTK is a teaching toolkit which is not really optimized for speed. Therefore, this may take forever. For speed, use scikit-learn for the classifiers.


accuracy(classifier, test_set)
classifier.classify(pos_features('cats'))


'''

To accompany the video, here is the sample code for NLTK part of speech tagging with lots of comments and info as well:

POS tag list:

CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''
'''
>>> nltk.corpus.brown.tagged_words()
[('The', 'AT'), ('Fulton', 'NP-TL'), ...]
>>> nltk.corpus.brown.tagged_words(tagset='universal')
[('The', 'DET'), ('Fulton', 'NOUN'), ...]
'''
t = "Cyprus, officially the Republic of Cyprus, is an island country in the Eastern Mediterranean and the third largest and third most populous island in the Mediterranean. Cyprus is located south of Turkey, west of Syria and Lebanon, northwest of Israel, north of Egypt, and southeast of Greece. Cyprus is a major tourist destination in the Mediterranean. With an advanced, high-income economy and a very high Human Development Index, the Republic of Cyprus has been a member of the Commonwealth since 1961 and was a founding member of the Non-Aligned Movement until it joined the European Union on 1 May 2004. On 1 January 2008, the Republic of Cyprus joined the eurozone."

from nltk import sent_tokenize, word_tokenize
sentences = sent_tokenize(t.lower())
sentences

from nltk import sent_tokenize, word_tokenize
sentences = sent_tokenize(t.lower())
sentences

tokens = word_tokenize(sentences[2])
tokens

from nltk import pos_tag
tags = pos_tag(tokens)
tags

import nltk.help
nltk.help.upenn_tagset('NN')


####similar words   wordnet

from nltk.corpus import wordnet as wn
wn.synsets('human')
wn.synsets('human')[0].definition()
wn.synsets('human')[1].definition()

human = wn.synsets('Human', pos=wn.NOUN)[0]
human


human.hypernyms() # A hypernym is a word with a broad meaning constituting a category into which words with more specific meanings fall; a superordinate. For example, colour is a hypernym of red.



bike = wn.synsets('bicycle')[0]
bike
girl = wn.synsets('girl')[1]
girl
###word similarity index
bike.wup_similarity(human)

girl.wup_similarity(human)


man = wn.synsets('man')[0]
woman= wn.synsets('woman')[0]
man.wup_similarity(woman)



synonyms = []
for syn in wn.synsets('girl'):
    print(syn)
    for lemma in syn.lemmas(): #  A lemma is basically the dictionary form or base form of a word, as opposed to the various inflected forms of a word. 
        print(lemma)
        synonyms.append(lemma.name())
synonyms


antonyms = []
for syn in wn.synsets("girl"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
antonyms


###chunking####
from nltk import pos_tag
tags = pos_tag(tokens)
tags

from nltk.chunk import RegexpParser
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunker = RegexpParser(grammar)
result = chunker.parse(tags)
result

chunker = RegexpParser(grammar)
result = chunker.parse(tags)
result