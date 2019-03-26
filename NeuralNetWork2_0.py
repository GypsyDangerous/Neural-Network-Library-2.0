import numpy as np 
import time
import os
from Layer import *
from loss import *

'''
helper functions to display the accuracy of the network as a percentage with two decimals
'''
def truncate(x, level=100):
	return int(x*level)/level

def percent(x, total=100, level=100):
	return truncate((x/total)*100, level)

class NeuralNetwork:

	def __init__(self, layers=list, activations=list, learning_rate=.001, epochs=1, loss=mse):
		'''
		parameters and hyperparameters for the network.
		layerinfo is a list of numbers that tells the network how many
		neurons go in each layer.
		activations is a list of functions that tells the network which activation function
		each layer uses.
		size is the number of layers (including the input layer)
		loss is the loss function that will be used in backpropagation
		'''
		self.layerinfo = layers
		self.activations = activations
		self.size = len(self.layerinfo)
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.loss = loss
		self.layers = []
		self.outputs = []
		self.accuracy = 0
		
		# initialize the networks layers
		for i in range(self.size):
			size = self.layerinfo[i]
			prevsize = self.layerinfo[i-1]
			acitivation = self.activations[i]
			
			if i == 0:
				self.layers.append(layer(size, size, acitivation))
			else:
				self.layers.append(layer(prevsize, size, acitivation))

 	# functions to adjust the hyperparameters of the network

	def setLearningRate(self, val):
		self.learning_rate = val

	def getLearningRate(self):
		return self.learningRate

	def dec_learningRate(self, inc=.00001):
		self.learningRate -= inc

	def setEpochs(self, val):
		self.epochs = val

	def getEpochs(self):
		return self.epochs

	def incEpochs(self, inc=1000):
		self.epochs+=inc

	# Work in Progress
	def mutate(mr):
		mutation_rate = mr



	def fit(self, inputArray, labels):
		'''
		train the network over all the training data in a random order for self.epochs number 
		of times
		'''
		datasize = len(inputArray)
		indices = np.arange(datasize)
		for i in range(self.epochs):
			np.random.shuffle(indices)
			for j in range(datasize):
				index = indices[j]
				info = inputArray[index]
				goal = labels[index]
				self.train(info, goal)
	

	def process(self, inputs):
		'''
		use the feedforward algorithm to get a guess from the network
		'''
		for i in range(self.size-1):
			if i == 0:
				inputlayer = self.layers[0]
				inputlayer.values = np.array(inputs).reshape(inputlayer.nodes, 1)
				
			self.layers[i].feedforward(self.layers[i+1]) # see layers class feedforward function
		self.outputs = self.layers[-1].values
		return self.outputs


	def test(self, test_data, test_labels):
		'''
		test the network over all the test data and print the accuracy
		'''
		datalen = len(test_data)
		acc = 0
		for i in range(datalen):
			data = test_data[i]
			label = test_labels[i]
			guess = self.process(data)
			Guess = np.argmax(guess)
			Label = np.argmax(label)
			error = self.loss(label, guess)
			if Guess == Label:
				acc += 1
			print(i, " error: %f, guess: %d, answer: %d" % (error, Guess, Label)) # optional line for debugging
		self.accuracy = percent(acc, datalen)
		print(self.accuracy, "%  accurate")


	def save(self, filename='2model.npz'):
		'''
		save all the important parameters and hyperparameters to a numpy zip file,
		if you dont have the "models" folder in your directory you will get an error
		'''
		np.savez_compressed(
			file = os.path.join(os.curdir, 'models', filename),
			layerinfo=self.layerinfo,
			activations=self.activations,
			size=self.size,
			loss=self.loss,
			layers=self.layers,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            )


	def load(self, filename='2model.npz'):
		'''
		load the saved parameters and hyperparameters to the network
		'''
		npz_members = np.load(os.path.join(os.curdir, 'models', filename))
		self.lauerinfo = (npz_members['layerinfo'])
		self.activations = (npz_members['activations'])
		self.size = (npz_members['size'])
		# self.loss = (npz_members['loss']) get an error when trying to load the loss function
		self.layers = (npz_members['layers'])
		self.learning_rate = (npz_members['learning_rate'])
		self.epochs = int(npz_members['epochs'])

	
	def train(self, inputs, targets):
		'''
		attempting to implement backpropagation with gradient descent. I need help with this part
		'''
		targets = np.array(targets)
		targets = targets.reshape(self.layers[-1].nodes, 1)
		guess = self.process(inputs)
		# for i in range()

	