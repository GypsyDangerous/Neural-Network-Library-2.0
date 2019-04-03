from mnistdata import *
from NeuralNetWork2_0 import *

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)

'''
list of integers defining the number of neurons in each layer of the network, index 0 or the first item of the list is the input layer so it should be the number of inputs you plan to send into the network
index -1 or the last item of the list is the output layer so it should be the expected outputs
'''
layers = [784, 256, 256, 10]
'''
list of activation functions which are assigned to the layer cooresponding their index, i.e the second function at index 1 will be assigned to layer 1 or the first hidden layer.
You can put any activation in index 0 because it will assigned to the input 'layer' and will not actually be used.
'''
activations = [inputs, tanh, tanh, softmax]

brain = NeuralNetwork(layers, activations, epochs = 4, loss = cross_entropy)
# brain.load()
brain.fit(trainingImages, trainingLabels, testImages, testLabels, True)
brain.save()


