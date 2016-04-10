"""
argv[1] = train_file
argv[2] = test_file
argv[3] = j

Also Note $ signs in POS tags was changed manually to character D, as vectorizers remove special characters
"""
import util
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sys import argv
import operator

train_x = []

train_x_MU = [[],[]]
train_x_SUS = [[],[]]
train_x_SUD = [[],[]]
train_x_PUS = [[],[]]
train_x_PUD = [[],[]]

train_x_MU_POS = [[],[]]
train_x_SUS_POS = [[],[]]
train_x_SUD_POS = [[],[]]
train_x_PUS_POS = [[],[]]
train_x_PUD_POS = [[],[]]

train_y = []

#Input training data
with open("../data/train.txt",'r') as f: 
	for line in f:
		a = util.processDatum(line)
		# train_x_MU.append(a['main_utterance'])
		if(a['label']==1):
			train_x_MU[1].append(a['main_utterance'])
			b = a['main_utterance']
			if('subsequent_utterance_same' in a.keys()):
				train_x_SUS[1].append(a['subsequent_utterance_same'])
				b = b +' '+ a['subsequent_utterance_same']
			else:
				train_x_SUD[1].append(a['subsequent_utterance_diff'])
				b = b +' '+ a['subsequent_utterance_diff']
			if('previous_utterance_same' in a.keys()):
				train_x_PUS[1].append(a['previous_utterance_same'])
				b = b +' '+ a['previous_utterance_same']
			else:
				train_x_PUD[1].append(a['previous_utterance_diff'])
				b = b +' '+ a['previous_utterance_diff']
			
		else:
			train_x_MU[0].append(a['main_utterance'])
			b = a['main_utterance']
			if('subsequent_utterance_same' in a.keys()):
				train_x_SUS[0].append(a['subsequent_utterance_same'])
				b = b +' '+ a['subsequent_utterance_same']
			else:
				train_x_SUD[0].append(a['subsequent_utterance_diff'])
				b = b +' '+ a['subsequent_utterance_diff']
			if('previous_utterance_same' in a.keys()):
				train_x_PUS[0].append(a['previous_utterance_same'])
				b = b +' '+ a['previous_utterance_same']
			else:
				train_x_PUD[0].append(a['previous_utterance_diff'])
				b = b +' '+ a['previous_utterance_diff']
		train_x.append(b)
		train_y.append(a['label'])
		
i = 0		
with open("../data/train_POS.txt",'r') as f: 
	for line in f:
		a = util.processPOSDatum(line.strip())
		b = train_x[i]
		if(train_y[i]  == 1):
			train_x_MU_POS[1].append(a['main_utterance_POS'])
			b = b +' '+ a['main_utterance_POS']
			if('subsequent_utterance_same_POS' in a.keys()):
				train_x_SUS_POS[1].append(a['subsequent_utterance_same_POS'])
				b = b +' '+ a['subsequent_utterance_same_POS']
			else:
				train_x_SUD_POS[1].append(a['subsequent_utterance_diff_POS'])
				b = b +' '+ a['subsequent_utterance_diff_POS']
			if('previous_utterance_same_POS' in a.keys()):
				train_x_PUS_POS[1].append(a['previous_utterance_same_POS'])
				b = b +' '+ a['previous_utterance_same_POS']
			else:
				train_x_PUD_POS[1].append(a['previous_utterance_diff_POS'])
				b = b +' '+ a['previous_utterance_diff_POS']
		else:
			train_x_MU_POS[0].append(a['main_utterance_POS'])
			b = b +' '+ a['main_utterance_POS']
			if('subsequent_utterance_same_POS' in a.keys()):
				train_x_SUS_POS[0].append(a['subsequent_utterance_same_POS'])
				b = b +' '+ a['subsequent_utterance_same_POS']
			else:
				train_x_SUD_POS[0].append(a['subsequent_utterance_diff_POS'])
				b = b +' '+ a['subsequent_utterance_diff_POS']
			if('previous_utterance_same_POS' in a.keys()):
				train_x_PUS_POS[0].append(a['previous_utterance_same_POS'])
				b = b +' '+ a['previous_utterance_same_POS']
			else:
				train_x_PUD_POS[0].append(a['previous_utterance_diff_POS'])
				b = b +' '+ a['previous_utterance_diff_POS']
		train_x[i] = b
		i = i + 1

test_x = []
test_y = []

# print train_x[0],train_x_MU[0][0]
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

def defaultGet(vocab,key):
	a = vocab.get(key)
	if(a is None):
		return 0
	return a

def vocabs(trainset,k,maxf,vocab,i):
	'''adds terms in vocabulary whose count should be maintained.'''
	
	tset = []
	tset.extend(trainset[0])
	tset.extend(trainset[1])
	cc =  ' '.join(tset)
	tvect = TfidfVectorizer(decode_error='ignore',ngram_range=(k,k),max_features=None)
	tvect.fit(tset)
	tfidf = tvect.transform([cc]) 
	
	tterms = tvect.get_feature_names()
	tfreqs = tfidf.sum(axis=0).A1
	tresult = dict(zip(tterms, tfreqs))

	
	#0
	cc0 =  ' '.join(trainset[0])
	cvect0 = CountVectorizer(decode_error='ignore',ngram_range=(k,k),max_features=None)
	cvect0.fit(trainset[0])
	count0 = cvect0.transform([cc0]) 
	
	terms0 = cvect0.get_feature_names()
	freqs0 = count0.sum(axis=0).A1
	result0 = dict(zip(terms0, freqs0))
	
	#1
	cc1 =  ' '.join(trainset[1])
	cvect1 = CountVectorizer(decode_error='ignore',ngram_range=(k,k),max_features=None)
	cvect1.fit(trainset[1])
	count1 = cvect1.transform([cc1])
	
	terms1 = cvect1.get_feature_names()
	freqs1 = count1.sum(axis=0).A1
	result1 = dict(zip(terms1, freqs1))

	result = {}
	terms = list(set(terms0)|set(terms1))
	ctr = 0
	for x in terms:
		a = defaultGet(result0,x)
		b = defaultGet(result1,x)
		c = max(a,b)
		result[x] = (float(c)/float(a+b))*tresult[x]
		if(result[x]==1):
			ctr = ctr +1
		# result[x] = (float(c)/float(a+b))



	sorted_result = sorted(result.items(), key=operator.itemgetter(1))
	# print len(sorted_result),ctr,sorted_result[0],sorted_result[-1]
	# print result
	feats = 0
	if maxf is not None:
		feats = max(0,(len(sorted_result)-maxf))
	for x,y in sorted_result[feats:]:
		if(x not in vocab.keys()):
			vocab[x] = i
			i = i + 1
	return (vocab,i)

vocab = {}
vocab['rprprprprprprprpprprprprprprpr']=0 #SVMLight does not have 0 index
i = 1
max_feats = None #1000
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
vect = TfidfVectorizer(decode_error='ignore',ngram_range=(1,1),vocabulary=vocab)
svm_train_x = vect.fit_transform(train_x)
train_file = '../svm/data.train'
if(len(argv)>2):
	train_file = argv[1]
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

