from sklearn.externals import joblib
import sys,subprocess,re, string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
import shlex,time


sys.path.append("src/")
import util

#############################################################
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
	affirm_list = ["yes","no","yeah","nah","right","wrong","absolutely"] 
	if any(i in affirm_list for i in utter.lower().split()):
		return True
	return False

###############################################################




#vectorizer: to extract features
vect = joblib.load('vect.pkl')

def dialog(main_utterance,previous_utterance,prev_same,succeeding_utterance,succ_same):
	if main_utterance == '':
		main_utterance = 't_empty'
	if previous_utterance == '':
		previous_utterance = 't_empty'
	if succeeding_utterance == '':
		succeeding_utterance = 't_empty'

	dialog_string = main_utterance
	dialog_string_POS = " ".join([pos for (w,pos) in util.givePOStags(main_utterance)])
	temp = ''
	temp_POS = ''
	if prev_same == 'y':
		print "X:",previous_utterance
		temp = ' &-1 ' + previous_utterance
		temp_POS = ' &-1 ' + " ".join([pos for (w,pos) in util.givePOStags(previous_utterance)])
	else:
		print "Y:",previous_utterance
		temp = ' %-1 ' + previous_utterance
		temp_POS = ' %-1 ' + " ".join([pos for (w,pos) in util.givePOStags(previous_utterance)])
	print "X:",main_utterance
	if succ_same == 'y':
		print "X:",succeeding_utterance
		dialog_string += ' & ' + succeeding_utterance
		dialog_string_POS += ' & ' + " ".join([pos for (w,pos) in util.givePOStags(succeeding_utterance)])
	else:
		print "Y:",succeeding_utterance
		dialog_string += ' % ' + succeeding_utterance
		dialog_string_POS += ' % ' + " ".join([pos for (w,pos) in util.givePOStags(succeeding_utterance)])

	dialog_string+=temp + ' &-2 t_empty'
	dialog_string_POS+=temp_POS + ' &-2 #'
	#add dummy label
	return ('0,' + dialog_string,'0,' + dialog_string_POS) 

def extractFeatures(dialog_string,dialog_string_POS):
	test_x = []
	test_y = []

	a = util.processDatum(dialog_string)
	b = a['main_utterance']
	test_y.append(a['label'])

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

	#ADD POS
	a = util.processPOSDatum(dialog_string_POS)
	b = b +' '+ a['main_utterance_POS']
	if('subsequent_utterance_same_POS' in a.keys()):
		b = b +' '+ a['subsequent_utterance_same_POS']
	else:
		b = b +' '+ a['subsequent_utterance_diff_POS']
	if('previous_utterance_same_POS' in a.keys()):
		b = b +' '+ a['previous_utterance_same_POS']
	else:
		b = b +' '+ a['previous_utterance_diff_POS']
	
	test_x.append(b)
	svm_test_x = vect.transform(test_x)
	dump_svmlight_file(svm_test_x,test_y,'demo.test')




print "############################################################"
print "############ Welcome to the demo of our project ############"
print "############################################################"

while True:
	print ""
	print "Enter the question you wish our system to judge:   (main-utterance)"
	main_utterance = raw_input()
	print "Enter the utterance previous to the main-utterance:(prev-utterance)"
	previous_utterance = raw_input()
	print "Is the prev-utterance spoken by the same person?   (y/n)"
	prev_same = raw_input()
	print "Enter the utterance succeeding the main-utterance: (succ-utterance)"
	succeeding_utterance = raw_input()
	print "Is the succ-utterance spoken by the same person?   (y/n)"
	succ_same = raw_input()

	print "#################### The Given Dialog ######################"
	dialog_string,dialog_string_POS = dialog(main_utterance,previous_utterance,prev_same,succeeding_utterance,succ_same)
	print "############################################################"



	#dumps the dialog to demo.test
	extractFeatures(dialog_string,dialog_string_POS)
	#run the classifier
	command = '"lib/svm ubuntu/svm_classify" demo.test clf output.txt'
	proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)


	print "##################### Our Prediction #######################"
	time.sleep(2)
	fo = open('output.txt')
	line = fo.readline()
	fo.close()
	if float(line) > 0:
		print "YES! It's a rhetorical question"
	else:
		print "NO! It's not a rhetorical question"
	print "############################################################"


