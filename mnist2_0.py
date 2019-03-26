from mnistdata import *
from NeuralNetWork2_0 import *

layers = [784, 256, 256, 256, 256, 10]
activations = [sigmoid, tanh, tanh, tanh, tanh, softmax]

trainingImages, trainingLabels = loadMNIST("train", "C:/Users/david/Downloads/datasets/mnist")
testImages, testLabels = loadMNIST("t10k", "C:/Users/david/Downloads/datasets/mnist")

trainingLabels = toHotEncoding(trainingLabels)
testLabels = toHotEncoding(testLabels)


brain = NeuralNetwork(layers, activations, loss=cross_entropy)
brain.load()
brain.test(testImages, testLabels)
brain.save()

