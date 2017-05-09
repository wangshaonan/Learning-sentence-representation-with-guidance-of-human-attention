import sys
import cPickle as pickle
import numpy as np
import math
from scipy.stats import pearsonr
import ppdb_utils

(vocab, We) = ppdb_utils.getWordmap('paragram-phrase-XXL.txt')
pos_vocab = ppdb_utils.getPosVocab('data.pos')
pp = np.load('word-pos-model.pkl')
embedding = pp[0].get_value()
pos_embedding = pp[1].get_value()

test0 = np.zeros((1000,80), dtype=int)
test0pos = np.zeros((1000,80), dtype=int)
mask0 = np.zeros((1000,80), dtype=float)
test1 = np.zeros((1000,80), dtype=int)
test1pos = np.zeros((1000,80), dtype=int)
mask1 = np.zeros((1000,80),dtype=float)
aNonTokens = ['.', "''", '``', ',', ':', ';', '?', '!', '-', '_', '(', ')']
test_score = []
numi = 0
#pos
for line in open(sys.argv[1]):
    line = line.lower()
    line = line.strip().split('\t')
    if not len(line) == 3:
        continue
    numj = 0
    for i in line[1].split():
        if not len(i.split('_')) == 3:
            continue
        word = i.split('_')[0]
	pos = i.split('_')[1]
        if word in vocab and pos in pos_vocab and word not in aNonTokens:
            test0[numi][numj] = int(vocab[word])
            test0pos[numi][numj] = int(pos_vocab[pos])
            mask0[numi][numj] = 1
	    numj += 1
    numj = 0
    for i in line[2].split():
        if not len(i.split('_')) == 3:
            continue
        word = i.split('_')[0]
        pos = i.split('_')[1]
        if word in vocab and pos in pos_vocab and word not in aNonTokens:
            test1[numi][numj] = int(vocab[word])
            test1pos[numi][numj] = int(pos_vocab[pos])
            mask1[numi][numj] = 1
            numj += 1
    test_score.append(float(line[0]))
    numi += 1

#embedding
emb1 = embedding[test0]
pos1 = pos_embedding[test0pos]
#dotsum
dot1 = (emb1*pos1).sum(axis=-1)
#softmax
tmp = np.exp(dot1)*mask0
soft1 = (tmp/tmp.sum(axis=1).reshape(1000,1)+1)
#mul
att1 = soft1.reshape(1000,80,1)*emb1  #1000*80*300
#average
sums1 = att1.sum(axis=1)
averages1 = sums1/(mask0.sum(axis=1)).reshape(1000,1)
#np.savetxt('ave1', averages1)

#embedding
emb2 = embedding[test1]
pos2 = pos_embedding[test1pos]
#dotsum
dot2 = (emb2*pos2).sum(axis=-1)
#softmax
tmp = np.exp(dot2)*mask1
soft2 = (tmp/tmp.sum(axis=1).reshape(1000,1)+1)
#mul
att2 = soft2.reshape(1000,80,1)*emb2  #1000*80*300
#average
sums2 = att2.sum(axis=1)
averages2 = sums2/(mask1.sum(axis=1)).reshape(1000,1)
#np.savetxt('ave2', averages2)

p1p2 = (averages1[0:len(test_score)]*averages2[0:len(test_score)]).sum(axis=1)
#np.savetxt('p1p2', p1p2)
p1p2norm = np.sqrt((averages1[0:len(test_score)] ** 2).sum(axis=1)) * np.sqrt((averages2[0:len(test_score)] ** 2).sum(axis=1))
cos = (p1p2/p1p2norm)  #cosine
cos = list(cos)
#np.savetxt('cos', cos)
del_ind = []
for i,c in enumerate(cos):
    if math.isnan(c):
#       print i
        del_ind.append(i)
sdel_ind = sorted(del_ind, reverse=True)
for d in sdel_ind:
    del test_score[d]
    del cos[d]
corr = pearsonr(cos, test_score)

print sys.argv[1]+'\t'+str(corr[0])
