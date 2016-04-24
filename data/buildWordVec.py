import gensim
import os,ast

class MySentences(object):
		def __init__(self, fname):
			self.fname = fname
	
		def __iter__(self):
			for line in open( self.fname):
				yield line
	
sentences = MySentences('train_for_brown.txt') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences,workers=4)
model.save('word2vec_model')
# model.most_similar(positive=['woman', 'king'], negative=['man'])
# print model['Urgent']
new_model = gensim.models.Word2Vec.load("word2vec_model")
# print new_model['2bhk']