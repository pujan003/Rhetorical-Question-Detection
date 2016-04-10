import classifier
import util
import scipy.sparse as sp
import numpy as np

test = []
with open("../data/test.txt",'r') as f: 
	for line in f:
		a = util.processDatum(line)
		test.append((a,a['label']))
