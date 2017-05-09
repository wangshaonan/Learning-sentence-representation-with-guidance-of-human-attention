import lasagne
import numpy as np
import theano.tensor as T
import sys


class gateLayer(lasagne.layers.MergeLayer):
  def __init__(self, incomings, **kwargs):
    super(gateLayer, self).__init__(incomings, **kwargs)

  def get_output_shape_for(self, input_shapes):
    return input_shapes[1]

  def get_output_for(self, inputs, **kwargs):
    return T.where(T.eq(inputs[0],0), np.float32(0.0), np.float32(1.0)).dimshuffle((0,1,'x')) * inputs[1]

class DotSumLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(DotSumLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return (input[0]*input[1]).sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

class softMaxLayer(lasagne.layers.MergeLayer):
  def __init__(self, incoming, **kwargs):
    super(softMaxLayer, self).__init__(incoming, **kwargs)

  def get_output_shape_for(self, input_shapes):
    '''
    The input is just a vector of numbers.
    The output is also a vector, same size as the input.
    '''
    return input_shapes[1]

  def get_output_for(self, inputs, **kwargs):
    '''
    Take the exp() of all inputs, and divide by the total.
    '''
    exps = T.where(T.eq(inputs[0],0), np.float32(0.0), np.float32(1.0)) * T.exp(inputs[1])

    return exps / (exps.sum(axis=1).dimshuffle((0, 'x')) + 1e-6)

class softMaxLayer2(lasagne.layers.MergeLayer):
  def __init__(self, incoming, **kwargs):
    super(softMaxLayer2, self).__init__(incoming, **kwargs)

  def get_output_shape_for(self, input_shapes):
    '''
    The input is just a vector of numbers.
    The output is also a vector, same size as the input.
    '''
    return input_shapes[1]

  def get_output_for(self, inputs, **kwargs):
    '''
    Take the exp() of all inputs, and divide by the total.
    '''
    tmp_mask = T.where(T.eq(inputs[0],0), np.float32(0.0), np.float32(1.0))
    exps = tmp_mask * T.exp(inputs[1])
    nums = tmp_mask.sum(axis=1)

    return ( exps / (exps.sum(axis=1).dimshuffle((0, 'x')) + 1e-6) )*nums.dimshuffle((0, 'x'))

class MulLayer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(MulLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return input[0].reshape((input[0].shape[0], input[0].shape[1], 1)) * input[1]

    def get_output_shape_for(self, input_shape):
        return input_shape[1]

class averageLayer(lasagne.layers.Layer):
  def __init__(self, incoming, fGradientClippingBound=None, **kwargs):
    super(averageLayer, self).__init__(incoming, **kwargs)

    self.fGradientClippingBound = fGradientClippingBound

  def get_output_shape_for(self, input_shape):
    '''
    The input is a batch of word vectors.
    The output is a single vector, same size as the input word embeddings
    In other words, since we are averaging, we loose the penultimate dimension
    '''
    return (input_shape[0], input_shape[2])

  def get_output_for(self, input, **kwargs):
    # Sums of word embeddings (so the zero embeddings don't matter here)
    sums = input.sum(axis=1)

    #normalisers = T.neq((T.neq(input, 0.0)).sum(axis=2, dtype='int32'), 0.0).sum(axis=1, dtype='floatX').reshape((-1, 1))

    #averages = sums / normalisers
    averages = sums

    return averages
