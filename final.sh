# Make features, train, validate and test
cd src
# $1 is the svm_file_name
python main.py $1
cd ..
python findTradeOff.py $1 >> $1_evals.txt
echo -e '\n' >> $1_evals.txt