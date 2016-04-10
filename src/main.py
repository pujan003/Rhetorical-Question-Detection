"""
argv[1] = train_file
argv[2] = test_file
argv[3] = j
"""
import util
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sys import argv
train_x = []

train_x_MU = []
train_x_SUS = []
train_x_SUD = []
train_x_PUS = []
train_x_PUD = []

train_x_MU_POS = []
train_x_SUS_POS = []
train_x_SUD_POS = []
train_x_PUS_POS = []
train_x_PUD_POS = []

train_y = []

#Input training data
with open("../data/train.txt",'r') as f: 
	for line in f:
		a = util.processDatum(line)
		train_x_MU.append(a['main_utterance'])
		b = a['main_utterance']
		if('subsequent_utterance_same' in a.keys()):
			train_x_SUS.append(a['subsequent_utterance_same'])
			b = b +' '+ a['subsequent_utterance_same']
		else:
			train_x_SUD.append(a['subsequent_utterance_diff'])
			b = b +' '+ a['subsequent_utterance_diff']
		if('previous_utterance_same' in a.keys()):
			train_x_PUS.append(a['previous_utterance_same'])
			b = b +' '+ a['previous_utterance_same']
		else:
			train_x_PUD.append(a['previous_utterance_diff'])
			b = b +' '+ a['previous_utterance_diff']
		train_y.append(a['label'])
		train_x.append(b)

i = 0		
with open("../data/train_POS.txt",'r') as f: 
	for line in f:
		a = util.processPOSDatum(line.strip())
		b = train_x[i]
		train_x_MU_POS.append(a['main_utterance_POS'])
		b = b +' '+ a['main_utterance_POS']
		if('subsequent_utterance_same_POS' in a.keys()):
			train_x_SUS_POS.append(a['subsequent_utterance_same_POS'])
			b = b +' '+ a['subsequent_utterance_same_POS']
		else:
			train_x_SUD_POS.append(a['subsequent_utterance_diff_POS'])
			b = b +' '+ a['subsequent_utterance_diff_POS']
		if('previous_utterance_same_POS' in a.keys()):
			train_x_PUS_POS.append(a['previous_utterance_same_POS'])
			b = b +' '+ a['previous_utterance_same_POS']
		else:
			train_x_PUD_POS.append(a['previous_utterance_diff_POS'])
			b = b +' '+ a['previous_utterance_diff_POS']
		train_x[i] = b
		i = i + 1

test_x = []
test_y = []

#input test data
with open("../data/test.txt",'r') as f: 
	for line in f:
		a = util.processDatum(line)
		b = a['main_utterance']
		if('subsequent_utterance_same' in a.keys()):
			b = b +' '+ a['subsequent_utterance_same']
		else:
			b = b +' '+ a['subsequent_utterance_diff']
		if('previous_utterance_same' in a.keys()):
			b = b +' '+ a['previous_utterance_same']
		else:
			b = b +' '+ a['previous_utterance_diff']
		test_y.append(a['label'])
		test_x.append(b)

i = 0		
with open("../data/test_POS.txt",'r') as f: 
	for line in f:
		a = util.processPOSDatum(line.strip())
		b = test_x[i]
		b = b +' '+ a['main_utterance_POS']
		if('subsequent_utterance_same_POS' in a.keys()):
			b = b +' '+ a['subsequent_utterance_same_POS']
		else:
			b = b +' '+ a['subsequent_utterance_diff_POS']
		if('previous_utterance_same_POS' in a.keys()):
			b = b +' '+ a['previous_utterance_same_POS']
		else:
			b = b +' '+ a['previous_utterance_diff_POS']
		test_x[i] = b
		i = i + 1


def vocabs(trainset,k,maxf,vocab,i):
	'''adds terms in vocabulary whose count should be maintained.'''
	cc =  ' '.join(trainset)
	cvect = CountVectorizer(decode_error='ignore',ngram_range=(k,k),max_features=maxf)
	cvect.fit(trainset)
	count = cvect.transform([cc]) 
	# print count
	it =  cvect.inverse_transform(count)
	it =  list(set(np.concatenate((tuple(it)))))
	# print it
	for x in it:
		if(x not in vocab.keys()):
			vocab[x] = i
			i = i + 1
	return (vocab,i)

vocab = {}
vocab['rprprprprprprprpprprprprprprpr']=0
i = 1
max_feats = None
if(len(argv) >= 4):
	max_feats = int(argv[3])

vocab,i = vocabs(train_x_MU,1,max_feats,vocab,i)
vocab,i = vocabs(train_x_MU,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUS,1,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUD,1,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUD,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUS,1,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUD,1,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUD,2,max_feats,vocab,i)

vocab,i = vocabs(train_x_MU_POS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_MU_POS,3,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUS_POS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUS_POS,3,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUD_POS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_SUD_POS,3,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUS_POS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUS_POS,3,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUD_POS,2,max_feats,vocab,i)
vocab,i = vocabs(train_x_PUD_POS,3,max_feats,vocab,i)

print "VOCABULARY MADE! ",len(vocab)
#Final feature Builder
vect = TfidfVectorizer(decode_error='ignore',ngram_range=(1,3),max_features=None,vocabulary=vocab)
vect.fit(train_x)
svm_train_x = vect.transform(train_x)
train_file = '../svm/data.train'
if(len(argv)>2):
	test_file = argv[1]
dump_svmlight_file(svm_train_x,train_y,train_file)

svm_test_x = vect.transform(test_x)
test_file = '../svm/data.test'
if(len(argv)>3):
	test_file = argv[2]
dump_svmlight_file(svm_test_x,test_y,test_file)

print "DONE!"

# it =  vect.inverse_transform(svm_train_x)
# it =  list(set(np.concatenate((tuple(it)))))
# for x in it:
# 	print x

