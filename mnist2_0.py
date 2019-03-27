from mnistdata import *
from NeuralNetWork2_0 import *
import time
import os

'''
functions from mnistdata module that load the mnist data from your directory, keep the first part of the function the same as it is
and change the second part to your directory.
'''
trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")


#functions from mnistdata module for getting the labels to one hot encoding

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

'''
list of integers defining the number of neurons in each layer of the network, 
index 0 or the first item of the list is the input layer so it should be the number of inputs you plan to send into the network
index -1 or the last item of the list is the output layer so it should be the expected outputs
'''
layers = [784, 256, 256, 256, 256, 10]
'''
list of activation functions which are assigned to the layer cooresponding their index, 
i.e the second function at index 1 will be assigned to layer 1 or the first hidden layer.
You can put any activation function in index 0 or the first space because it will assigned to the input "layer" and will not actually be used.
'''
activations = [sigmoid, tanh, tanh, tanh, tanh, softmax]

brain = NeuralNetwork(layers, activations, loss=cross_entropy)
# brain.load()
brain.test(testImages, testLabels)
# brain.save()


