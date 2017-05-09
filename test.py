import numpy as np

pp = np.load('sick-sts-word-pos-model.pkl')
print pp
print pp[0].get_value().shape
print pp[1].get_value().shape
