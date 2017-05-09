import lasagne

class lasagne_average_layer(lasagne.layers.MergeLayer):
    
    def __init__(self, incoming, **kwargs):
        super(lasagne_average_layer, self).__init__(incoming, **kwargs)
    
    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
	att = inputs[1]
	mask = inputs[2]

	emb = (att*emb).sum(axis=1)
        #emb = emb / mask.sum(axis=1)[:, None]
        return emb
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],input_shape[0][2])
