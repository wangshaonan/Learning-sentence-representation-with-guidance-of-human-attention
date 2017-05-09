import theano
import numpy as np
from theano import tensor as T
from theano import config
import lasagne
from lasagne_attention_layer import gateLayer
from lasagne_attention_layer import DotSumLayer
from lasagne_attention_layer import softMaxLayer
from lasagne_attention_layer import MulLayer
from lasagne_attention_layer import averageLayer

import cPickle

class word_model_sim(object):

    def getRegTerm(self, params, We, initial_We):
        l2 = 0.5*params.LC*sum(lasagne.regularization.l2(x) for x in self.network_params)
        if params.traintype == "normal":
            if params.updatewords:
                return l2 + 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
            else:
                return l2
        elif params.traintype == "reg":
            return l2 + 0.5*params.LRW*lasagne.regularization.l2(We-initial_We)
        elif params.traintype == "rep":
            return l2
        else:
            raise ValueError('Params.traintype not set correctly.')

    def getTrainableParams(self, params):
        if params.traintype == "rep":
            return self.network_params
        if params.updatewords or params.traintype == "reg":
            return self.all_params
        else:
            return self.network_params

    def __init__(self, We_initial, We_pos_initial, params):

        if params.maxval:
            self.nout = params.maxval - params.minval + 1

        if params.traintype == "reg" or params.traintype == "rep":
            p = cPickle.load(file(params.regfile, 'rb'))
            print p #containes We

        if params.traintype == "reg":
            print "regularizing to parameters"

        if params.traintype == "rep":
            print "not updating embeddings"

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
	We_pos = theano.shared(np.asarray(We_pos_initial, dtype = config.floatX))

        if params.traintype == "reg":
            initial_We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))

        if params.traintype == "rep":
            We = theano.shared(np.asarray(p[0].get_value(), dtype = config.floatX))

        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
	g1posbatchindices = T.imatrix(); g2posbatchindices = T.imatrix()
        scores = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None, 1)) #25*50
	l_pos = lasagne.layers.InputLayer((None, None, 1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)  #25*50*300
	l_pos_emb = lasagne.layers.EmbeddingLayer(l_pos, input_size=We_pos.get_value().shape[0], output_size=We_pos.get_value().shape[1], W=We_pos)
	#mask
        llGate = gateLayer([l_in, l_emb], name='llGate') #25*50*300

  	#attention-vector
  	llDot = DotSumLayer([llGate, l_pos_emb], name='llDot') #25*50
  	llSoftMax = softMaxLayer([l_in, llDot], name='llSoftMax') #25*30 mask
	#llSoftMax_out = lasagne.layers.get_output(llSoftMax, {l_in:g1batchindices, l_pos:g1posbatchindices})
	#self.look = theano.function([g1batchindices,g1posbatchindices], llSoftMax_out)

  	llAttend = MulLayer([llSoftMax, llGate], name='llAttend') #25*50*300
  	llAverage = averageLayer(llAttend, fGradientClippingBound=1, name='llAverage')  #25*300

        embg1 = lasagne.layers.get_output(llAverage, {l_in:g1batchindices, l_pos:g1posbatchindices})
        embg2 = lasagne.layers.get_output(llAverage, {l_in:g2batchindices, l_pos:g2posbatchindices})

        g1_dot_g2 = embg1*embg2
        g1_abs_g2 = abs(embg1-embg2)

        lin_dot = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        lin_abs = lasagne.layers.InputLayer((None, We.get_value().shape[1]))
        l_sum = lasagne.layers.ConcatLayer([lin_dot, lin_abs])
        l_sigmoid = lasagne.layers.DenseLayer(l_sum, params.memsize, nonlinearity=lasagne.nonlinearities.sigmoid)

        l_softmax = lasagne.layers.DenseLayer(l_sigmoid, self.nout, nonlinearity=T.nnet.softmax)
	
        X = lasagne.layers.get_output(l_softmax, {lin_dot:g1_dot_g2, lin_abs:g1_abs_g2})
        Y = T.log(X)

        cost = scores*(T.log(scores) - Y)
        cost = cost.sum(axis=1)/(float(self.nout))

        prediction = 0.
        i = params.minval
        while i<= params.maxval:
            prediction = prediction + i*X[:,i-1]
            i += 1

        self.network_params = lasagne.layers.get_all_params(llAverage, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)
        self.network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(llAverage, trainable=True) + lasagne.layers.get_all_params(l_softmax, trainable=True)

        reg = self.getRegTerm(params, We, initial_We)
        self.trainable = self.getTrainableParams(params)
        cost = T.mean(cost) + reg

        self.feedforward_function = theano.function([g1batchindices,g1posbatchindices], embg1)
        self.scoring_function = theano.function([g1batchindices, g1posbatchindices,g2batchindices,g2posbatchindices],prediction)
        self.cost_function = theano.function([scores, g1batchindices,g1posbatchindices,g2batchindices,g2posbatchindices], cost)

        grads = theano.gradient.grad(cost, self.trainable)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.trainable, params.eta)
        self.train_function = theano.function([scores, g1batchindices, g1posbatchindices, g2batchindices, g2posbatchindices], cost, updates=updates)
