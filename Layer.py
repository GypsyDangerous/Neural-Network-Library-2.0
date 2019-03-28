import numpy as np 
from activations import *

class layer:
	'''
	a simple layer class to hold wieghts, biases, and an activation function
	'''
	def __init__(self, inputs, nodes, activation):
		'''
		all the parameters of a layer.
		the inputs is the a number that representing the size of the previous layer 
		or the number of inputs to the network.
		the nodes is the number that represents the size of the layer.
		the activation is a function that is the layers activation function
		the weights is a matrix of weights for each neuron in the layer
		the biases is a vector of biases for each neuron in the layer
		the values is a vector that holds the value of each neuron in the layer,
		they are what is matrix multiplied with the next layers weights to get the weighted sum
		'''
		self.inputs = inputs
		self.nodes = nodes
		self.activation = activation
		self.activation_p = activation
		
		if self.activation == sigmoid:
			self.activation_p = sigmoid_p
		elif self.activation == softmax or self.activation == stable_softmax:
			self.activation_p = softmax_p
		elif self.activation == tanh:
			self.activation_p = tanh_p
		elif self.activation == relu:
			self.activation_p = relu_p

		self.weights = 2*np.random.rand(self.nodes, self.inputs)-1
		self.biases = 2*np.random.rand(self.nodes, 1)-1
		self.values = np.zeros(self.nodes)
		self.values = self.values.reshape(self.nodes, 1)


	def feedforward(self, n):
			'''
			matrix multiply this layers values with the next layers weights, add the next layers biases
			and send it through the next layers activation function. 
			'''
			n.values = n.activation(np.dot(n.weights, self.values)+(n.biases))
