import numpy as np

'''
a helper module for loss functions, more are going to be added.
'''

def cross_entropy(targets, guess):
	'''
	targets is a one hot encoded vector and guess is the output vector of the network
	'''
	return np.sum(targets * (-np.log(guess)))

def cross_entropy_p(targets, guess):
	return -np.sum(targets / guess)


def mse(targets, guess):
	return np.mean((targets-guess)**2)

def mse_p(targets, guess):
	return guess - targets

