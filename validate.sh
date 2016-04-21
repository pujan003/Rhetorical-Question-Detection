lib/svm\ ubuntu/svm_learn -j 15 -c $3 svm/$1 clf >/dev/null
lib/svm\ ubuntu/svm_classify  -v 3 svm/$2 clf output
