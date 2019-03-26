import numpy as np

'''
a helper module for loss functions, more are going to be added.
'''

def cross_entropy(targets, guess):
	entropy = 0
	for i in range(len(targets)):
		if targets[i] == 1:
			entropy -= np.log(guess[i])
		else:
			entropy -= np.log(1 - guess[i])
	return entropy[0]


def mse(targets, guess):
		return np.sum((targets-guess)**2)/len(targets)