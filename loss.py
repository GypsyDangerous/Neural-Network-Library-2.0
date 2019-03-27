import numpy as np

'''
a helper module for loss functions, more are going to be added.
'''

def cross_entropy(targets, guess):
	return np.sum(targets * -np.log(guess))

def cross_entropy_p(targets, guess):
	return -np.sum(targets / guess)


def mse(targets, guess):
		return np.mean((targets-guess)**2)

def mse_p(targets, guess):
	return guess - targets


if __name__ == "__main__":
	test = np.array([0, 0, 0, 1, 0, 0])
	other = np.array([0.01, 0, 0, 0.9, 0, 0])
	test = test.reshape(6, 1)
	other = other.reshape(6, 1)
	print(other)
	print(cross_entropy(test, other))
