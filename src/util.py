"""
Holds all the utility functions used in the project.
A function 'f' can be used anywhere as util.f() in the project by importing util.py
"""
import nltk
from nltk.tag import pos_tag


def giveTokens(sentence):
	'''Returns a list of tokens'''
	return nltk.word_tokenize(sentence)

def givePOStags(sentence,isSentenceTokenized = False):
	'''Returns a list of (word,tag) given a sentence or a tokenized sentence'''
	tokens = sentence
	if not(isSentenceTokenized): 
		try:
			tokens = text = nltk.word_tokenize(sentence)
		except TypeError:
			print "WARN: Given sentence is not a string. Trying to treat it as tokens"
	posTagged = nltk.pos_tag(tokens)
	return posTagged
