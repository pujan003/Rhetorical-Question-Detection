lib/svm\ ubuntu/svm_learn -j 15 -c $2 svm/$1.train clf
lib/svm\ ubuntu/svm_classify  -v 3 svm/$1.test clf output
