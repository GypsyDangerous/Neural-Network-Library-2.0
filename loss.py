import numpy as np
from activations import *

'''
a helper module for loss functions, more are going to be added.
'''

def cross_entropy(targets, guess):
	return np.sum(targets * -np.log(guess))

# I don't know if this is correct but it is what I got from https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py
def cross_entropy_p(targets, guess):
	return guess - targets 


def mse(targets, guess):
		return np.mean((targets-guess)**2)

def mse_p(targets, guess):
	return guess - targets

