# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:54:27 2020

@author: anshu
"""
##F9 only for selected cell
##ctrl+4 to comment
##ctrl+5 to uncomment



import spacy
nlp = spacy.load('en')
##nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

# Print each token separately
for token in doc:
    print(token.text, token.pos_, token.dep_)
    
print(nlp.pipeline)

##tokenization 
doc2 = nlp(u"Tesla isn't   looking into startups anymore.")

print(doc2[0])



##check dependencies of token
##full dependency list https://spacy.io/api/annotation#dependency-parsing
print(doc2[0].dep_)



# =============================================================================
# =============================================================================
# # .text
# # The original word text
# # Tesla
# # .lemma_
# # The base form of the word
# # tesla
# # .pos_
# # The simple part-of-speech tag
# # PROPN/proper noun
# # .tag_
# # The detailed part-of-speech tag
# # NNP/noun, proper singular
# # .shape_
# # The word shape – capitalization, punctuation, digits
# # Xxxxx
# # .is_alpha
# # Is the token an alpha character?
# # True
# # .is_stop
# # Is the token part of a stop list, i.e. the most common words of the language?
# # False
# # */
# # =============================================================================
# 
# =============================================================================

# Lemmas (the base form of the word):
print(doc2[4].text)
print(doc2[4].lemma_)

# Simple Parts-of-Speech & Detailed Tags:
print(doc2[4].pos_)
print(doc2[4].tag_ + ' / ' + spacy.explain(doc2[4].tag_))

# Word Shapes:
print(doc2[0].text+': '+doc2[0].shape_)
print(doc[5].text+' : '+doc[5].shape_)

# Boolean Values:
print(doc2[0].is_alpha)
print(doc2[0].is_stop)


doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')
    
print(doc3[1])

##span  is slice of Doc object in the form Doc[start:stop].

for token in doc3:
    print (token.i,token.text)
    
life_quote = doc3[16:30]
print(life_quote)

type(life_quote)


#####SENTENCES########
doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
for sent in doc4.sents:
    print(sent)
    
doc4[6].is_sent_start

###TOKENIZATION####
# Create a string that includes opening and closing quotation marks
mystring = '"We\'re moving to L.A.!"'
print(mystring)

###marking tokens ending with  |
# Create a Doc object and explore tokens
doc = nlp(mystring)

for token in doc:
    print(token.text, end=' | ')

doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")

for t in doc4:
    print(t)
    
len(doc)

##Counting vocab entries
len(doc.vocab)

###retrieving tokens from Text

doc5 = nlp(u'It is better to give than to receive.')

# Retrieve the third token:
doc5[2]

# Retrieve three tokens from the middle:
doc5[2:5]

# Retrieve the last four tokens:
doc5[-4:]

##REMEMBER TypeError: 'spacy.tokens.doc.Doc' object does not support item assignment

###NER
doc8 = nlp(u'Apple to build a Hong Kong factory for $6 million')

for token in doc8:
    print(token.text, end=' | ')

print('\n----')

for ent in doc8.ents:
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    
##Extract number of entities
len(doc8.ents)

###named entities visit https://spacy.io/usage/linguistic-features#named-entities

##Noun chunks
doc9 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc9.noun_chunks:
    print(chunk.text)
    
##noun_chunks visit https://spacy.io/usage/linguistic-features#noun-chunks
    
##Displacy For more info visit https://spacy.io/usage/visualizers


###stemming is not avaliable in spacy but available in nltk
# Import the toolkit and the full Porter Stemmer library
import nltk

from nltk.stem.porter import *

p_stemmer = PorterStemmer()
words = ['run','runner','running','ran','runs','easily','fairly']
for word in words:
    print(word+' --> '+p_stemmer.stem(word))
    
## Snowball stemmer  It offers a slight improvement over the original Porter stemmer, both in logic and speed.
from nltk.stem.snowball import SnowballStemmer

# The Snowball Stemmer requires that you pass a language parameter
s_stemmer = SnowballStemmer(language='english')

words = ['run','runner','running','ran','runs','easily','fairly']
for word in words:
    print(word+' --> '+s_stemmer.stem(word))
    
###Lemmatization in spacy

doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")

for token in doc1:
    print(token.text, '\t', token.pos_, '\t',  token.lemma_)
    
##display neatly
##Since the display above is staggared and hard to read, let's write a function that displays the information we want more neatly.
    
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma_}')

doc2 = nlp(u"I saw eighteen mice today!")

show_lemmas(doc2)


###stopwords
# Print the set of spaCy's default stop words (remember that sets are unordered):
print(nlp.Defaults.stop_words)

len(nlp.Defaults.stop_words)

nlp.vocab['myself'].is_stop
nlp.vocab["next"].is_stop
print(nlp.vocab)

# Add the word to the set of stop words. Use lowercase!
nlp.Defaults.stop_words.add('btw')

# Set the stop_word tag on the lexeme
nlp.vocab['btw'].is_stop = True

# Remove the word from the set of stop words
nlp.Defaults.stop_words.remove('beyond')

# Remove the stop_word tag from the lexeme
nlp.vocab['beyond'].is_stop = False



# =============================================================================
# Rule-based Matching
# spaCy offers a rule-matching tool called Matcher that allows you to build a library of token patterns, then match those patterns against a Doc object to return a list of found matches. You can match on any part of the token including text and annotations, and you can add multiple patterns to the same matcher.
# 
# =============================================================================

# Import the Matcher library
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)


##creating pattern
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

###applying matcher to doc
doc = nlp(u'The Solar Power industry continues to grow as demand \
for solarpower increases. Solar-power cars are gaining popularity.')

found_matches = matcher(doc)
print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start:end]                    # get the matched span
    print(match_id, string_id, start, end, span.text)
    
##You can make token rules optional by passing an 'OP':'*' argument. This lets us streamline our patterns list
# Redefine the patterns:
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)

found_matches = matcher(doc)
print(found_matches)

# =============================================================================
# The following quantifiers can be passed to the 'OP' key:
# OP
# Description
# \!
# Negate the pattern, by requiring it to match exactly 0 times
# ?
# Make the pattern optional, by allowing it to match 0 or 1 times
# \+
# Require the pattern to match 1 or more times
# \*
# Allow the pattern to match zero or more times
# =============================================================================

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}] # CHANGE THIS PATTERN

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)

doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')

found_matches = matcher(doc2)
print(found_matches)


pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solarpowered'}]
pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'powered'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2, pattern3, pattern4)

found_matches = matcher(doc2)
print(found_matches)

# =============================================================================
# 
# Other token attributes
# Besides lemmas, there are a variety of token attributes we can use to determine matching rules:
# Attribute
# Description
# `ORTH`
# The exact verbatim text of a token
# `LOWER`
# The lowercase form of the token text
# `LENGTH`
# The length of the token text
# `IS_ALPHA`, `IS_ASCII`, `IS_DIGIT`
# Token text consists of alphanumeric characters, ASCII characters, digits
# `IS_LOWER`, `IS_UPPER`, `IS_TITLE`
# Token text is in lowercase, uppercase, titlecase
# `IS_PUNCT`, `IS_SPACE`, `IS_STOP`
# Token is punctuation, whitespace, stop word
# `LIKE_NUM`, `LIKE_URL`, `LIKE_EMAIL`
# Token text resembles a number, URL, email
# `POS`, `TAG`, `DEP`, `LEMMA`, `SHAPE`
# The token's simple and extended part-of-speech tag, dependency label, lemma, shape
# `ENT_TYPE`
# The token's entity label
# =============================================================================


##Token wildcard
# =============================================================================
# You can pass an empty dictionary {} as a wildcard to represent any token. For example, you might want to retrieve hashtags without knowing what might follow the # character:
# [{'ORTH': '#'}, {}]
# =============================================================================




# =============================================================================
# PhraseMatcher
# In the above section we used token patterns to perform rule-based matching. An alternative - and often more efficient - method is to match on terminology lists. In this case we use PhraseMatcher to create a Doc object from a list of phrases, and pass that into matcher instead.
# =============================================================================

# Import the PhraseMatcher library
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
# Print the full text:
print(doc.text)

# Print the fifth word and associated tags:
print(doc[4].text, doc[4].pos_, doc[4].tag_, spacy.explain(doc[4].tag_))

for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

# =============================================================================
# Every token is assigned a POS Tag from the following list:
# POS
# DESCRIPTION
# EXAMPLES
# ADJ
# adjective
# *big, old, green, incomprehensible, first*
# ADP
# adposition
# *in, to, during*
# ADV
# adverb
# *very, tomorrow, down, where, there*
# AUX
# auxiliary
# *is, has (done), will (do), should (do)*
# CONJ
# conjunction
# *and, or, but*
# CCONJ
# coordinating conjunction
# *and, or, but*
# DET
# determiner
# *a, an, the*
# INTJ
# interjection
# *psst, ouch, bravo, hello*
# NOUN
# noun
# *girl, cat, tree, air, beauty*
# NUM
# numeral
# *1, 2017, one, seventy-seven, IV, MMXIV*
# PART
# particle
# *'s, not,*
# PRON
# pronoun
# *I, you, he, she, myself, themselves, somebody*
# PROPN
# proper noun
# *Mary, John, London, NATO, HBO*
# PUNCT
# punctuation
# *., (, ), ?*
# SCONJ
# subordinating conjunction
# *if, while, that*
# SYM
# symbol
# *$, %, §, ©, +, −, ×, ÷, =, :), 😝*
# VERB
# verb
# *run, runs, running, eat, ate, eating*
# X
# other
# *sfpksdpsxmsa*
# SPACE
# space
# =============================================================================

##POS-For a current list of tags for all languages visit https://spacy.io/api/annotation#pos-tagging

doc = nlp(u'I read a book on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

###counting the POS tags of doc 
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# Count the frequencies of different coarse-grained POS tags:
POS_counts = doc.count_by(spacy.attrs.POS)
POS_counts

doc.vocab[83].text

for k,v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')
    
# Count the different fine-grained tags:
TAG_counts = doc.count_by(spacy.attrs.TAG)

for k,v in sorted(TAG_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{4}}: {v}')
    
# Count the different dependencies in document:
DEP_counts = doc.count_by(spacy.attrs.DEP)

for k,v in sorted(DEP_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{4}}: {v}')
    
# =============================================================================
# These are some grammatical examples (shown in bold) of specific fine-grained tags. We've removed punctuation and rarely used tags:
# POS
# TAG
# DESCRIPTION
# EXAMPLE
# ADJ
# AFX
# affix
# The Flintstones were a **pre**-historic family.
# ADJ
# JJ
# adjective
# This is a **good** sentence.
# ADJ
# JJR
# adjective, comparative
# This is a **better** sentence.
# ADJ
# JJS
# adjective, superlative
# This is the **best** sentence.
# ADJ
# PDT
# predeterminer
# Waking up is **half** the battle.
# ADJ
# PRP$
# pronoun, possessive
# **His** arm hurts.
# ADJ
# WDT
# wh-determiner
# It's blue, **which** is odd.
# ADJ
# WP$
# wh-pronoun, possessive
# We don't know **whose** it is.
# ADP
# IN
# conjunction, subordinating or preposition
# It arrived **in** a box.
# ADV
# EX
# existential there
# **There** is cake.
# ADV
# RB
# adverb
# He ran **quickly**.
# ADV
# RBR
# adverb, comparative
# He ran **quicker**.
# ADV
# RBS
# adverb, superlative
# He ran **fastest**.
# ADV
# WRB
# wh-adverb
# **When** was that?
# CONJ
# CC
# conjunction, coordinating
# The balloon popped **and** everyone jumped.
# DET
# DT
# determiner
# **This** is **a** sentence.
# INTJ
# UH
# interjection
# **Um**, I don't know.
# NOUN
# NN
# noun, singular or mass
# This is a **sentence**.
# NOUN
# NNS
# noun, plural
# These are **words**.
# NOUN
# WP
# wh-pronoun, personal
# **Who** was that?
# NUM
# CD
# cardinal number
# I want **three** things.
# PART
# POS
# possessive ending
# Fred**'s** name is short.
# PART
# RP
# adverb, particle
# Put it **back**!
# PART
# TO
# infinitival to
# I want **to** go.
# PRON
# PRP
# pronoun, personal
# **I** want **you** to go.
# PROPN
# NNP
# noun, proper singular
# **Kilroy** was here.
# PROPN
# NNPS
# noun, proper plural
# The **Flintstones** were a pre-historic family.
# VERB
# MD
# verb, modal auxiliary
# This **could** work.
# VERB
# VB
# verb, base form
# I want to **go**.
# VERB
# VBD
# verb, past tense
# This **was** a sentence.
# VERB
# VBG
# verb, gerund or present participle
# I am **going**.
# VERB
# VBN
# verb, past participle
# The treasure was **lost**.
# VERB
# VBP
# verb, non-3rd person singular present
# I **want** to go.
# VERB
# VBZ
# verb, 3rd person singular present
# He **wants** to go.
# =============================================================================

###NER explaination####
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')

doc = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')

show_ents(doc)

# =============================================================================
# Doc.ents are token spans with their own set of annotations.
# `ent.text`
# The original entity text
# `ent.label`
# The entity type's hash value
# `ent.label_`
# The entity type's string description
# `ent.start`
# The token span's *start* index position in the Doc
# `ent.end`
# The token span's *stop* index position in the Doc
# `ent.start_char`
# The entity text's *start* index position in the Doc
# `ent.end_char`
# The entity text's *stop* index position in the Doc
# =============================================================================

doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')

for ent in doc.ents:
    print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)
    
# =============================================================================
# Tags are accessible through the .label_ property of an entity.
# TYPE
# DESCRIPTION
# EXAMPLE
# `PERSON`
# People, including fictional.
# *Fred Flintstone*
# `NORP`
# Nationalities or religious or political groups.
# *The Republican Party*
# `FAC`
# Buildings, airports, highways, bridges, etc.
# *Logan International Airport, The Golden Gate*
# `ORG`
# Companies, agencies, institutions, etc.
# *Microsoft, FBI, MIT*
# `GPE`
# Countries, cities, states.
# *France, UAR, Chicago, Idaho*
# `LOC`
# Non-GPE locations, mountain ranges, bodies of water.
# *Europe, Nile River, Midwest*
# `PRODUCT`
# Objects, vehicles, foods, etc. (Not services.)
# *Formula 1*
# `EVENT`
# Named hurricanes, battles, wars, sports events, etc.
# *Olympic Games*
# `WORK_OF_ART`
# Titles of books, songs, etc.
# *The Mona Lisa*
# `LAW`
# Named documents made into laws.
# *Roe v. Wade*
# `LANGUAGE`
# Any named language.
# *English*
# `DATE`
# Absolute or relative dates or periods.
# *20 July 1969*
# `TIME`
# Times smaller than a day.
# *Four hours*
# `PERCENT`
# Percentage, including "%".
# *Eighty percent*
# `MONEY`
# Monetary values, including unit.
# *Twenty Cents*
# `QUANTITY`
# Measurements, as of weight or distance.
# *Several kilometers, 55kg*
# `ORDINAL`
# "first", "second", etc.
# *9th, Ninth*
# `CARDINAL`
# Numerals that do not fall under another type.
# *2, Two, Fifty-two*
# =============================================================================
    
doc = nlp(u'Tesla to build a U.K. factory for $6 million')

show_ents(doc)

#Right now, spaCy does not recognize "Tesla" as a company.
from spacy.tokens import Span

# Get the hash value of the ORG entity label
ORG = doc.vocab.strings[u'ORG']  

# Create a Span for the new entity
new_ent = Span(doc, 0, 1, label=ORG)

# Add the entity to the existing Doc object
doc.ents = list(doc.ents) + [new_ent]

show_ents(doc)

####adding entities using phrase matcher
doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
          u'If successful, the vacuum cleaner will be our first product.')

show_ents(doc)
# Import PhraseMatcher and create a matcher object:
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Create the desired phrase patterns:
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]


# Apply the patterns to our matcher object:
matcher.add('newproduct', None, *phrase_patterns)

# Apply the matcher to our Doc object:
matches = matcher(doc)

# See what matches occur:
matches

# Here we create Spans from each match, and create named entities from them:
from spacy.tokens import Span

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = [Span(doc, match[1],match[2],label=PROD) for match in matches]

doc.ents = list(doc.ents) + new_ents

show_ents(doc)

doc2 = nlp("Microsoft is a company")
print(doc2.ents)

##counting entities
doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')

show_ents(doc)
len([ent for ent in doc.ents if ent.label_=='MONEY'])


spacy.__version__


##adding pipleine to nlp pipeline
# Quick function to remove ents formed on whitespace:
def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc

# Insert this into the pipeline AFTER the ner component:
nlp.add_pipe(remove_whitespace_entities, after='ner')

# Rerun nlp on the text above, and show ents:
doc = nlp(u'Originally priced at $29.50,\nthe sweater was marked down to five dollars.')


show_ents(doc)



##For more on Named Entity Recognition visit https://spacy.io/usage/linguistic-features#101

# =============================================================================
# noun_chunks components:
# `.text`
# The original noun chunk text.
# `.root.text`
# The original text of the word connecting the noun chunk to the rest of the parse.
# `.root.dep_`
# Dependency relation connecting the root to its head.
# `.root.head.text`
# The text of the root token's head.
# =============================================================================

doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc.noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)

print(len(list(doc.noun_chunks)))


#For more on noun_chunks visit https://spacy.io/usage/linguistic-features#noun-chunks


doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
doc_sents = [sent for sent in doc.sents]
doc_sents

# Now you can access individual sentences:
print(doc_sents[1])

print(doc_sents[1].start, doc_sents[1].end)

##adding rules
# Parsing the segmentation start tokens happens during the nlp pipeline
doc2 = nlp(u'This is a sentence. This is a sentence. This is a sentence.')

for token in doc2:
    print(token.is_sent_start, ' '+token.text)
    
# SPACY'S DEFAULT BEHAVIOR
doc3 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc3.sents:
    print(sent)
    
# ADD A NEW RULE TO THE PIPELINE
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc

# Re-run the Doc object creation:
doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc4.sents:
    print(sent)

nlp.add_pipe(set_custom_boundaries, before='parser')

nlp.pipe_names

doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc4.sents:
    print(sent)
    
#In this section we'll see how the default sentencizer breaks on periods. We'll then replace this behavior with a sentencizer that breaks on linebreaks.
mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."

# SPACY DEFAULT BEHAVIOR:
doc = nlp(mystring)

for sent in doc.sents:
    print([token.text for token in sent])
# CHANGING THE RULES
from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'): # handles multiple occurrences
            seen_newline = True
    yield doc[start:]      # handles the last group of tokens


sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
nlp.add_pipe(sbd)

doc = nlp(mystring)
for sent in doc.sents:
    print([token.text for token in sent])