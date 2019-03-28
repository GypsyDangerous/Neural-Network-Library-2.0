import numpy as np

'''
simple helper module for nueral network activation functions
'''

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return x * (1-x)

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))

def stable_softmax(x):
	return np.exp(x-max(x))/np.sum(np.exp(x))

def softmax_p(x):
	return x * (1-x)

def tanh(x):
	return np.tanh(x)

def tanh_p(x):
	return 1 - x * x

def relu(x):
	return np.maximum(x, 0)

def relu_p(x):
	return float(x > 0)

def leaky_relu(x):
	return np.maximum(x, .1*x)

def leaky_relu_p(x):
	return float(x > .1)

def softplus(x):
	return np.log(1+np.exp(x))

def softplus_p(x):
	return sigmoid(x)

	



