import numpy as np 
import time
import os
from Layer import *
from loss import *
from misc import *



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
		self.loss_p = loss
		if self.loss == mse:
			self.loss_p = mse_p
		elif self.loss == cross_entropy:
			self.loss_p = cross_entropy_p
		
		# initialize the networks layers
		for i in range(self.size):
			size = self.layerinfo[i]
			prevsize = self.layerinfo[i-1]
			acitivation = self.activations[i]
			
			if i == 0:
				self.layers.append(layer(size, size, acitivation))
			else:
				self.layers.append(layer(prevsize, size, acitivation))

	'''functions to adjust the hyperparameters of the network'''

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


	def fit(self, trainingArray, labels, test_data=None, test_labels=None, test=False):

		# error checking
		if len(trainingArray) > len(labels):
			raise Exception("Your have more data points than labels")

		if len(trainingArray) < len(labels):
			raise Exception("Your have more label points than data points")


		'''
		train the network over all the training data in a random order for self.epochs number 
		of times
		'''
		datasize = len(trainingArray)
		indices = np.arange(datasize)
		start = time.time()
		if test:
			info = "Training, Testing mode: active"
		else:
			info = "Training, Testing mode: inactive"
		print(info)
		for i in range(self.epochs):
			np.random.shuffle(indices)
			for j in range(datasize):
				index = indices[j]
				info, goal = trainingArray[index], labels[index]

				# self.process(info)

				nabla_b = [np.zeros(self.layers[i].biases.shape) for i in range(self.size)]
				nabla_w = [np.zeros(self.layers[i].weights.shape) for i in range(self.size)]

				# delta_nabla_b, delta_nabla_w = self.backprop(goal)
				# nabla_w += delta_nabla_w
				# nabla_b += delta_nabla_b

				# for i in range(len(nabla_w)):
				# 	self.layers[i].update_weights(nabla_w[i], self.learning_rate)

				# for i in range(len(nabla_b)):
				# 	self.layers[i].update_biases(nabla_b[i], self.learning_rate)


			Epoch = i+1

			if not test:
				print("epoch: %d, %a%% finished" % (Epoch, percent(Epoch, self.epochs)))
			else:
				print("epoch: %d, Accuracy: %a%%, %a%% finished" % 
					(Epoch, self.test(test_data, test_labels), percent(Epoch, self.epochs)))

		end = time.time();
		print("Training complete, time passed:", time_format(end-start), )


	def process(self, inputs):
		'''
		use the feedforward algorithm to get a guess from the network
		'''
		for i in range(self.size-1):
			if i == 0:
				self.layers[0].input_layer(inputs)
				
			self.layers[i].feedforward(self.layers[i+1]) # see layers class feedforward function
		self.outputs = self.layers[-1].activations
		return self.outputs


	def test(self, test_data, test_labels, Print=False):

		# error checking
		if len(test_data) > len(test_labels):
			raise ValueError("Your have more test data points than labels")

		if len(test_data) < len(test_labels):
			raise ValueError("Your have more test labels than data points")


		'''
		test the network over all the test data and print the accuracy
		'''
		datalen = len(test_data)

		acc = 0
		for i in range(datalen):
			data = test_data[i]
			labellen = len(test_labels[i])
			label = test_labels[i].reshape(labellen, 1)
			guess = self.process(data)
			Guess = np.argmax(guess)
			Label = np.argmax(label)
		
			verdict = ""
			if Guess == Label:
				acc += 1
				verdict = "correct"
			else:
				verdict = "incorrect"
			if Print:
				error = self.loss(label, guess)
				confidence = percent(guess[Guess], 1, 100)
				print(i, " error: %f, guess: %d, answer: %d, confidence: %a%%, verdict %s" % (error, Guess, Label, confidence, verdict)) # optional line for debugging

		self.accuracy = percent(acc, datalen)
		if Print:
			print("accuracy: %a%%" % (self.accuracy))
		return self.accuracy



	def backprop(self, targets):
		'''
		backpropagation function im trying to make, there is a bug somewhere that I can't find, 
		it is causing matrix size errors in the dot product
		'''
		nabla_b = [np.zeros(self.layers[i].biases.shape) for i in range(self.size)]
		nabla_w = [np.zeros(self.layers[i].weights.shape) for i in range(self.size)]


		error = (self.loss_p(targets, self.outputs) * self.layers[-1].activation_p(self.layers[-1].outputs))

		nabla_b[-1] = error
		nabla_w[-1] = error.dot(self.layers[-2].activations.T)

		for i in range(self.size - 2, 1, -1):
			error = self.layers[i+1].weights.T.dot(error) * self.layers[i].activation_p(self.layers[i].outputs)
			nabla_b[i] = error
			nabla_w[i] = error.dot(self.layers[i-1].activations.T)

		return nabla_b, nabla_w


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

	





	
