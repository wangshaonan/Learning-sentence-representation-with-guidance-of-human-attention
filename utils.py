import theano
import numpy as np
from theano import config
from time import time
import cPickle
import sys
import cPickle as pkl

def checkIfQuarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

def prepare_data(list_of_seqs,list_of_pos):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_pos = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
	x_mask[idx, :lengths[idx]] = 1.
    for idx, s in enumerate(list_of_pos):
        x_pos[idx, :lengths[idx]] = s
    x_mask = np.asarray(x_mask, dtype=config.floatX)
    return x, x_pos, x_mask


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def getDataSim(batch, nout):
    g1, g1_pos = [], []
    g2, g2_pos = [], []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)
	g1_pos.append(i[0].pos_embeddings)
	g2_pos.append(i[1].pos_embeddings)

    g1x, g1x_pos, g1x_mask = prepare_data(g1, g1_pos)
    g2x, g2x_pos, g2x_mask = prepare_data(g2, g2_pos)

    scores = []
    for i in batch:
        temp = np.zeros(nout)
        score = float(i[2])
        ceil, fl = int(np.ceil(score)), int(np.floor(score))
        if ceil == fl:
            temp[fl - 1] = 1
        else:
            temp[fl - 1] = ceil - score
            temp[ceil - 1] = score - fl
        scores.append(temp)
    scores = np.matrix(scores) + 0.000001
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1x_pos, g1x_mask, g2x, g2x_pos, g2x_mask)

def getDataSim2(batch, nout):
    g1, g1_pos = [], []
    g2, g2_pos = [], []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)
        g1_pos.append(i[0].pos_embeddings)
        g2_pos.append(i[1].pos_embeddings)

    g1x, g1x_pos, g1x_mask = prepare_data(g1, g1_pos)
    g2x, g2x_pos, g2x_mask = prepare_data(g2, g2_pos)

    scores = []
    for i in batch:
        scores.append(float(i[2]))
    scores = np.asarray(scores, dtype=config.floatX)
    return (scores, g1x, g1x_pos, g1x_mask, g2x, g2x_pos, g2x_mask)

def getDataEntailment(batch):
    g1 = []; g2 = []
    g1_pos = []; g2_pos = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)
	g1_pos.append(i[0].pos_embeddings)
	g2_pos.append(i[1].pos_embeddings)

    g1x, g1x_pos = prepare_data(g1, g1_pos)
    g2x, g2x_pos = prepare_data(g2, g2_pos)

    scores = []
    for i in batch:
        temp = np.zeros(3)
        label = i[2].strip().lower()
        if label == "contradiction":
            temp[0]=1
        if label == "neutral":
            temp[1]=1
        if label == "entailment":
            temp[2]=1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype=config.floatX)
    return (scores,g1x,g1x_pos,g2x,g2x_pos)

def getDataParaphrase(batch):
    g1 = []; g2 = []
    g1_pos = []; g2_pos = []
    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)
        g1_pos.append(i[0].pos_embeddings)
        g2_pos.append(i[1].pos_embeddings)

    g1x, g1x_pos = prepare_data(g1, g1_pos)
    g2x, g2x_pos = prepare_data(g2, g2_pos)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[2].strip()
        if label == "0":
            temp[0]=1
        if label == "1":
            temp[1]=1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype=config.floatX)
    return (scores,g1x,g1x_pos,g2x,g2x_pos)

def getDataSentiment(batch):
    g1 = []
    g1_pos = []
    for i in batch:
        g1.append(i[0].embeddings)
	g1_pos.append(i[0].pos_embeddings)

    g1x, g1x_pos = prepare_data(g1,g1_pos)

    scores = []
    for i in batch:
        temp = np.zeros(2)
        label = i[1].strip()
        if label == "0":
            temp[0]=1
        if label == "1":
            temp[1]=1
        scores.append(temp)
    scores = np.matrix(scores)+0.000001
    scores = np.asarray(scores,dtype=config.floatX)
    return (scores,g1x,g1x_pos)

def train(model, train_data, words, pos_vocab, params):
        start_time = time()
	best_val = 0
	best_p = None
        try:
            for eidx in xrange(params.epochs):

                kf = get_minibatches_idx(len(train_data), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1
                    batch = [train_data[t] for t in train_index]

                    for i in batch:
                        i[0].populate_embeddings(words, pos_vocab)
                        i[1].populate_embeddings(words, pos_vocab)

                    (scores,g1x,g1x_pos,g1x_mask,g2x,g2x_pos,g2x_mask) = getDataSim(batch, model.nout)
                    cost = model.train_function(scores, g1x, g1x_pos, g2x, g2x_pos)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'

                    #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

                    #undo batch to save RAM
                    for i in batch:
                        i[0].representation = None
                        i[1].representation = None
                        i[0].unpopulate_embeddings()
                        i[1].unpopulate_embeddings()

		pkl.dump(model.all_params, open('%s.pkl'% params.outfile, 'wb'))
                print 'Epoch ', (eidx+1), 'Cost ', cost

                sys.stdout.flush()

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time()
        print "total time:", (end_time - start_time)
