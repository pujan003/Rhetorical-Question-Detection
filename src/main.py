"""
argv[1] = svm_file_name
argv[3] = j

Also Note $ signs in POS tags was changed manually to character D, as vectorizers remove special characters
"""
import util
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib

from sys import path
from sys import argv
import operator
import string
import random
from datetime import datetime

path.append("../lib/")
import gensim
word2vec_model = gensim.models.Word2Vec.load("../data/word2vec_model")
brown_cluster = {} # word --> [bit_string, cluster num]

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

train1_x = []
train1_y = []
validate_x = []
validate_y = []
positive_x = []
negative_x = []

def speakers(my_dict):
	s = ""
	if(my_dict['ss'] == 1):
		s = s + "ss1 "
	else:
		s = s + "ss0 "
	if(my_dict['sd'] == 1):
		s = s + "sd1 "
	else:
		s = s + "sd0 "
	if(my_dict['ps'] == 1):
		s = s + "ps1 "
	else:
		s = s + "ps0 "
	if(my_dict['pd'] == 1):
		s = s + "pd1 "
	else:
		s = s + "pd0 "
	return s.strip()

def isWHQ(utter):
	exclude = set(string.punctuation)
	utter = ''.join(ch for ch in utter if ch not in exclude)
	wh_list = ["who","what","how","which","why"] 
	if any(i in wh_list for i in utter.lower().split()):
		return True
	return False

def isAffirmative(utter):
	exclude = set(string.punctuation)
	utter = ''.join(ch for ch in utter if ch not in exclude)
	affirm_list = ["yes","no","yeah","nah","right","wrong","absolutely","hmmm"] 
	if any(i in affirm_list for i in utter.lower().split()):
		return True
	return False

def giveSimilarWords(x,top_k):
	try:
		similar_words = word2vec_model.most_similar(positive=[x], negative=[], topn=top_k)
	except KeyError:
		similar_words = []
	return similar_words

def loadBrownFile():
	f = open('../data/train_brown.txt')
	lines = f.readlines()
	for l in lines:
		[w,b,c] = l.strip().split()
		brown_cluster[w] = [b,c]

def assignBrown(x,j):
	ret = ''
	i = 0
	# for c in x:
	# 	ret+= c + 'POS'+str(i) + 'NUM'+ str(j) + ' '
	# 	i+=1
	ret = x[:7]
	return ret


loadBrownFile()

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
				## WHQ FEATURE
				if isWHQ(a['main_utterance']) and isAffirmative(a['subsequent_utterance_same']):
					b = b + ' ' + "WHQYES"
				b = b +' '+ a['subsequent_utterance_same']
			else:
				train_x_SUD[1].append(a['subsequent_utterance_diff'])
				## WHQ FEATURE
				if isWHQ(a['main_utterance']) and isAffirmative(a['subsequent_utterance_diff']):
					b = b + ' ' + "WHQYES"
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

		## BROWN FEATURE 
		dialog = a['main_utterance'].replace("?", " ?")
		dialog = dialog.replace("."," .")
		sen_pos = -1
		for token in dialog.split():
			sen_pos+=1
			try:
				b = b+' '+ assignBrown(brown_cluster[token][0],sen_pos)
				# print b
			except KeyError:
				continue

		b = b + " " + speakers(a)
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
		if train_y[i]  == 1:
			positive_x.append(b)
		else:
			negative_x.append(b)
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
			## WHQ FEATURE
			if isWHQ(a['main_utterance']) and isAffirmative(a['subsequent_utterance_same']):
				b = b + ' ' + "WHQYES"
			b = b +' '+ a['subsequent_utterance_same']
		else:
			## WHQ FEATURE
			if isWHQ(a['main_utterance']) and isAffirmative(a['subsequent_utterance_diff']):
				b = b + ' ' + "WHQYES"
			b = b +' '+ a['subsequent_utterance_diff']
		if('previous_utterance_same' in a.keys()):
			b = b +' '+ a['previous_utterance_same']
		else:
			b = b +' '+ a['previous_utterance_diff']
		b = b + " " + speakers(a)
		## BROWN FEATURE 
		dialog = a['main_utterance'].replace("?", " ?")
		dialog = dialog.replace("."," .")
		sen_pos = -1
		for token in dialog.split():
			sen_pos+=1
			try:
				b = b+' '+ assignBrown(brown_cluster[token][0],sen_pos)
			except KeyError:
				continue

		test_x.append(b)
		test_y.append(a['label'])
		

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
if(len(argv) >= 3):
	max_feats = int(argv[2])

# vocab,i = vocabs(train_x_MU,1,2000,vocab,i) #2000
# vocab,i = vocabs(train_x_MU,2,5000,vocab,i) #5000
# vocab,i = vocabs(train_x_SUS,1,500,vocab,i) #500
# vocab,i = vocabs(train_x_SUS,2,1000,vocab,i) #1000
# vocab,i = vocabs(train_x_SUD,1,1000,vocab,i) #1000
# vocab,i = vocabs(train_x_SUD,2,3000,vocab,i) #3000
# vocab,i = vocabs(train_x_PUS,1,500,vocab,i) #500
# vocab,i = vocabs(train_x_PUS,2,1000,vocab,i) #1000
# vocab,i = vocabs(train_x_PUD,1,1000,vocab,i) #1000
# vocab,i = vocabs(train_x_PUD,2,3000,vocab,i) #3000

# vocab,i = vocabs(train_x_MU_POS,2,500,vocab,i) #500-200
# vocab,i = vocabs(train_x_MU_POS,3,2000,vocab,i) #2000
# vocab,i = vocabs(train_x_SUS_POS,2,500,vocab,i) #500-200
# vocab,i = vocabs(train_x_SUS_POS,3,1000,vocab,i) #1000
# vocab,i = vocabs(train_x_SUD_POS,2,500,vocab,i) #500
# vocab,i = vocabs(train_x_SUD_POS,3,2000,vocab,i) #2000
# vocab,i = vocabs(train_x_PUS_POS,2,500,vocab,i) #500-200
# vocab,i = vocabs(train_x_PUS_POS,3,500,vocab,i) #500
# vocab,i = vocabs(train_x_PUD_POS,2,500,vocab,i) #500
# vocab,i = vocabs(train_x_PUD_POS,3,500,vocab,i) #500


vocab,i = vocabs(train_x_MU,1,316,vocab,i) #2000
vocab,i = vocabs(train_x_MU,2,742,vocab,i) #5000
vocab,i = vocabs(train_x_SUS,1,61,vocab,i) #500
vocab,i = vocabs(train_x_SUS,2,21,vocab,i) #1000
vocab,i = vocabs(train_x_SUD,1,484,vocab,i) #1000
vocab,i = vocabs(train_x_SUD,2,2167,vocab,i) #3000
vocab,i = vocabs(train_x_PUS,1,17,vocab,i) #500
vocab,i = vocabs(train_x_PUS,2,16,vocab,i) #1000
vocab,i = vocabs(train_x_PUD,1,36,vocab,i) #1000
vocab,i = vocabs(train_x_PUD,2,1141,vocab,i) #3000

vocab,i = vocabs(train_x_MU_POS,2,67,vocab,i) #500-200
vocab,i = vocabs(train_x_MU_POS,3,4118,vocab,i) #2000
vocab,i = vocabs(train_x_SUS_POS,2,54,vocab,i) #500-200
vocab,i = vocabs(train_x_SUS_POS,3,874,vocab,i) #1000
vocab,i = vocabs(train_x_SUD_POS,2,413,vocab,i) #500
vocab,i = vocabs(train_x_SUD_POS,3,40,vocab,i) #2000
vocab,i = vocabs(train_x_PUS_POS,2,36,vocab,i) #500-200
vocab,i = vocabs(train_x_PUS_POS,3,35,vocab,i) #500
vocab,i = vocabs(train_x_PUD_POS,2,539,vocab,i) #500
vocab,i = vocabs(train_x_PUD_POS,3,316,vocab,i) #500

vocab['ss0'] = i
i = i + 1 
vocab['ss1'] = i
i = i + 1 
vocab['sd0'] = i
i = i + 1 
vocab['sd1'] = i
i = i + 1 
vocab['ps0'] = i
i = i + 1 
vocab['ps1'] = i
i = i + 1 
vocab['pd0'] = i
i = i + 1 
vocab['pd1'] = i
i = i + 1
vocab['WHQYES'] = i
i = i + 1
# ADD HERE
# for k in xrange(0,50):
# 	for j in xrange(0,50):
# 		vocab['0POS'+str(k)+'NUM'+str(j)] = i
# 		vocab['1POS'+str(k)+'NUM'+str(j)] = i+1
# 		i+=2
# for k_ in brown_cluster.keys():
# 	k = k_[:7]
# 	if k not in vocab:
# 		vocab[k] = i
# 		i+=1
print "VOCABULARY MADE! ",len(vocab)

#Final feature Builder
vect = TfidfVectorizer(decode_error='ignore',ngram_range=(1,3),vocabulary=vocab)
# now you can save it to a file
# and later you can load it

svm_train_x = vect.fit_transform(train_x)
joblib.dump(vect, '../vect.pkl') 

train_file = '../svm/data.train'
if(len(argv)>1):
	train_file = '../svm/'+argv[1]+'.train'

dump_svmlight_file(svm_train_x,train_y,train_file)
print "Dumped to",train_file

svm_test_x = vect.transform(test_x)
test_file = '../svm/data.test'
if(len(argv)>1):
	test_file = '../svm/'+argv[1]+'.test'
# print svm_test_x
dump_svmlight_file(svm_test_x,test_y,test_file)
print "Dumped to",test_file

# #SPILT train_x and train_y into train1_x, validate_x train1_y, validate_y
pos_divide = int(len(positive_x)*0.8)
neg_divide = int(len(negative_x)*0.8)

SEED = datetime.now()
random.seed(SEED)
random.shuffle(positive_x)
random.shuffle(negative_x)
# print len(positive_x)
train1_x = positive_x[:pos_divide]
train1_y = [1]*len(train1_x)
train1_x += negative_x[:neg_divide]
train1_y += [-1]*len(negative_x[:neg_divide])

validate_x = positive_x[pos_divide:]
validate_y = [1]*len(positive_x[pos_divide:])
validate_x += negative_x[neg_divide:]
validate_y += [-1]*len(negative_x[neg_divide:])


svm_train1_x = vect.fit_transform(train1_x)
train1_file = '../svm/data.train1'
if(len(argv)>1):
	train1_file = '../svm/'+argv[1]+'.train1'
dump_svmlight_file(svm_train1_x,train1_y,train1_file)
print "Dumped to",train1_file

svm_validate_x = vect.transform(validate_x)
validate_file = '../svm/data.validate'
if(len(argv)>1):
	validate_file = '../svm/'+argv[1]+'.validate'
dump_svmlight_file(svm_validate_x,validate_y,validate_file)
print "Dumped to",validate_file



print "DONE!"


# it =  vect.inverse_transform(svm_train_x)
# it =  list(set(np.concatenate((tuple(it)))))
# for x in it:
# 	print x