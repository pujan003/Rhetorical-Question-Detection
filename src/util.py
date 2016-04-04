"""
Holds all the utility functions used in the project.
A function 'f' can be used anywhere as util.f() in the project by importing util.py
"""
import nltk
from nltk.tag import pos_tag
import re
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

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

def processDatum(datum):
	'''Returns a dictionary of extracted items given an entire data point i.e (label,dialog)
	Keys of the returned dict:
	label --> 0/1
	main_utterance --> some string
	subsequent_utterance_same --> string or this key is absent
	subsequent_utterance_diff --> string or this key is absent
	previous_utterance_same --> string or this key is absent
	previous_utterance_diff --> string or this key is absent
	previous_previous_utterance_same --> string or this key is absent
	previous_previous_utterance_diff --> string or this key is absent
	'''
	ret = {}
	label = int(datum.split(',')[0])
	ret['label'] = label
	dialog = ','.join(datum.split(',')[1:]) #safe
	
	ret['main_utterance'] = re.split('& | %',dialog)[0]
	
	subsequent_utterance = re.split('&-1 | %-1',re.split('& | %',dialog)[1])[0]
	if "& " in dialog:
		ret['subsequent_utterance_same'] = subsequent_utterance
	elif "% " in dialog:
		ret['subsequent_utterance_diff'] = subsequent_utterance
	else:
		print "ERROR: subsequent_utterance not found!"

	previous_utterance = re.split('&-2 | %-2',re.split('&-1 | %-1',dialog)[1])[0]
	if "&-1" in dialog:
		ret['previous_utterance_same'] = previous_utterance
	elif "%-1" in dialog:
		ret['previous_utterance_diff'] = previous_utterance
	else:
		print "ERROR: previous_utterance not found!"

	previous_previous_utterance = re.split('&-2 | %-2',dialog)[1]
	if "&-2" in dialog:
		ret['previous_previous_utterance_same'] = previous_previous_utterance
	elif "%-2" in dialog:
		ret['previous_previous_utterance_diff'] = previous_previous_utterance
	else:
		print "ERROR: previous_previous_utterance not found!"

	return ret


def giveEvaluations(Y_gold,Y_predicted):
	'''Returns a dictionary of evaluated measures'''
	accuracy = accuracy_score(Y_gold, Y_predicted)*100.0
	f1 = f1_score(Y_gold,Y_predicted)*100.0
	precision = precision_score(Y_gold,Y_predicted)*100.0
	recall = recall_score(Y_gold,Y_predicted)*100.0
	return {"Accuracy": accuracy, "F1":f1, "Precision":precision, "Recall":recall}