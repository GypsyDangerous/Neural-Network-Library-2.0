import numpy as np

'''
helper module for loading mnist data from the original ubyte files
''' 

# load the MNIST data from files
def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([ nImages, width, height ])

    labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]

    return data, labels



# convert the labels to hotencoding
def toHotEncoding(classification):
    # emulates the functionality of tf.keras.utils.to_categorical( y )
    hotEncoding = np.zeros([len(classification), 
                              np.max(classification) + 1])
    hotEncoding[np.arange(len(hotEncoding)), classification] = 1
    return hotEncoding