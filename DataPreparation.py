from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def create_serial_dataset(x=None, y=None, n=100, length=10, frames=10):

    X = np.zeros([n, length*frames, x.shape[1], x.shape[2]])
    Y = np.zeros([n, length*frames, 10])

    for i in range(n):

        k = np.random.randint(0, 1000, size=(length))
        for j in range(length):
          
            X[i, j*frames:j*frames+frames, :, :] = x[k[j], :, :]
            Y[i, j*frames-1:j*frames+frames-1, y[k[j]]] = 1
          
    return X.reshape(n, length*frames, x.shape[1] * x.shape[2]), Y

def init_MNIST():

    (x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
    Xn, Yn = create_serial_dataset(x_train, y_train, n=100, length=10, frames=10)
    Xn /= 255.0
    Yn /= 255.0
    X = Xn[:10, :, :]
    Y = Yn[:10, :, :]
    return X, Y, Xn, Yn
