import sys,subprocess,re

svm_file = sys.argv[1]

train_file = svm_file + '.train'
train1_file = svm_file + '.train1'
validation_file = svm_file + '.validate'
test_file = svm_file + '.test'

OPTIMIZE_ON_TEST = False
if "--test" in sys.argv:
	OPTIMIZE_ON_TEST = True
	validation_file = test_file
	train1_file = train_file

def f1(p,r):
	return 2*p*r/(p+r)

max_f1 = 0
max_c = 0
max_p = 0
max_r = 0
max_a = 0

for c in xrange(1,200):
	cost = c/100.0
	command = 'sh validate.sh '+train1_file+' '+validation_file+' '+str(cost)
	proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	tmp_ = proc.stdout.read()
	tmp = re.findall(r'\d+.\d+%', tmp_)
	# print tmp_
	try:
		a = float(tmp[0][:-1])
		p = float(tmp[1][:-1])
		r = float(tmp[2][:-1])
	except IndexError:
		continue

	f_1 = f1(p,r)
	# print "C=",cost," A=",a," P=",p," R=",r," F1=",f_1
	if f_1 > max_f1:
		max_f1 = f_1
		max_c = cost
		max_p = p
		max_r = r
		max_a = a

print "ON VALIDATION SET:"
print "Max F1 =",max_f1," for c =",max_c
print "P =",max_p," R =",max_r," A =",max_a

if "--test" not in sys.argv:
	print "ON TEST SET:"
	command = 'sh train_test.sh ' + svm_file +' '+str(max_c)
	proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
	tmp = re.findall(r'\d+.\d+%', proc.stdout.read())
	a = float(tmp[0][:-1])
	p = float(tmp[1][:-1])
	r = float(tmp[2][:-1])
	f_1 = f1(p,r)
	print "F1 =",f_1
	print "P =",p," R =",r," A =",a
