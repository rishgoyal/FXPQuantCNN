import numpy as np
from datasets.fashion_mnist.utils import mnist_reader
import keras


def get_data():

    x_train, y_train = mnist_reader.load_mnist('C:/Users/320060820/experiments/datasets/fashion_mnist/data/fashion', 
                                                kind='train')
    x_test, y_test = mnist_reader.load_mnist('C:/Users/320060820/experiments/datasets/fashion_mnist/data/fashion', 
                                                kind='t10k')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)