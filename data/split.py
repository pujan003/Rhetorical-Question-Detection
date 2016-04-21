import sys

ifile = open('train.txt')
ifile1 = open('train_POS.txt')

"""
80-20 split into train and validation set
"""


# ifile = open("a2_train.txt")
lines = ifile.readlines()
lines1 = ifile1.readlines()
pos_lines = []
pos_lines1 = []
neg_lines = []
neg_lines1 = []

for line in lines:
	label = line.split(',')[0]
	if label == "0":
		neg_lines.append(line)
	else:
		pos_lines.append(line)

for line in lines1:
	label = line.split(',')[0]
	if label == "0":
		neg_lines1.append(line)
	else:
		pos_lines1.append(line)

ofile1 = open('train1.txt',"wb")
ofile2 = open('validate1.txt',"wb")
ofile3 = open('train1_POS.txt',"wb")
ofile4 = open('validate1_POS.txt',"wb")

divide_pos = int(len(pos_lines)*0.8)
pos_lines_train1 = pos_lines[:divide_pos]
pos_lines_validate = pos_lines[divide_pos:]
pos_lines_train1_POS = pos_lines1[:divide_pos]
pos_lines_validate_POS = pos_lines1[divide_pos:]

divide_neg = int(len(neg_lines)*0.8)
neg_lines_train1 = neg_lines[:divide_neg]
neg_lines_validate = neg_lines[divide_neg:]
neg_lines_train1_POS = neg_lines1[:divide_neg]
neg_lines_validate_POS = neg_lines1[divide_neg:]


for l in pos_lines_train1:
	ofile1.write(l)
for l in pos_lines_train1_POS:
	ofile3.write(l)
for l in neg_lines_train1:
	ofile1.write(l)
for l in neg_lines_train1_POS:
	ofile3.write(l)

for l in pos_lines_validate:
	ofile2.write(l)
for l in neg_lines_validate:
	ofile2.write(l)
for l in pos_lines_validate_POS:
	ofile4.write(l)
for l in neg_lines_validate_POS:
	ofile4.write(l)


ofile2.close()
ofile1.close()
ifile.close()
ofile3.close()
ofile4.close()



