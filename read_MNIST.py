import struct
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

def mnist_data():
    DATASET_DIR = 'C:/data/'
    with gzip.open(os.path.join(DATASET_DIR,'train-images-idx3-ubyte.gz')) as f:
        f = f.read()
        magic, number, row, col = struct.unpack_from( '>IIII', f, 0)
        loaded = np.frombuffer(f, dtype=np.uint8)
        train_X = loaded[16:].reshape((number, 1, row, col))
    
    with gzip.open(os.path.join(DATASET_DIR,'train-labels-idx1-ubyte.gz')) as f:
        f = f.read()
        magic, number = struct.unpack_from( '>II', f, 0)
        loaded = np.frombuffer(f, dtype=np.uint8)
        train_Y = loaded[8:].reshape((number))
        
    with gzip.open(os.path.join(DATASET_DIR,'t10k-images-idx3-ubyte.gz')) as f:
        f = f.read()
        magic, number, row, col = struct.unpack_from( '>IIII', f, 0)
        loaded = np.frombuffer(f, dtype=np.uint8)
        test_X = loaded[16:].reshape((number, 1, row, col))
        
    with gzip.open(os.path.join(DATASET_DIR,'t10k-labels-idx1-ubyte.gz')) as f:
        f = f.read()
        magic, number = struct.unpack_from( '>II', f, 0)
        loaded = np.frombuffer(f, dtype=np.uint8)
        test_Y = loaded[8:].reshape((number))
        
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = mnist_data()

train_X = train_X.reshape(60000, 28*28)
test_X = test_X.reshape(10000, 28*28)

plt.imshow(train_X[0].reshape(28, 28))
plt.show()
