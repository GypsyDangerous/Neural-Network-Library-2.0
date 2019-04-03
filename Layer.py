import numpy as np 
from activations import *

class layer:
	'''
	a simple layer class to hold a wieght matrix, biase vector, and an activation function
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
		the outputs is a vector that holds the value of each neuron in the layer,
		the activations is a vector that holds the value of each neuron after going through an activation function
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
		elif self.activation == softplus:
			self.activation_p == softplus_p

		self.weights = 2*np.random.rand(self.nodes, self.inputs)-1
		self.biases = 2*np.random.rand(self.nodes, 1)-1

		self.outputs = np.zeros(self.nodes).reshape(self.nodes, 1)
		self.activations = np.zeros(self.nodes).reshape(self.nodes, 1)
		

	def input_layer(self, inputs):
		'''
		use on the input layer to send in the inputs to the network
		'''
		self.activations = np.array(inputs).reshape(self.nodes, 1)

	def feedforward(self, n):
			'''
			matrix multiply this layers outputs with the next layers weights, add the next layers biases
			and send it through the next layers activation function. 
			'''
			n.outputs = np.dot(n.weights, self.activations)+(n.biases)
			n.activations = n.activation(n.outputs)

	'''functions to update weights and biases from from backpropagation'''
	def update_weights(self, delta_weights, learning_rate=.01):
		self.weights -= (delta_weights*learning_rate)

	def update_biases(self, delta_biases, learning_rate=.01):
		self.biases -= (delta_biases*learning_rate)
