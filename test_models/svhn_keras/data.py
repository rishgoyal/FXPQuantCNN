import numpy as np
import keras
import scipy.io as sio
import os

def get_data():

    train_data = sio.loadmat('C:/Users/320060820/experiments/datasets/svhn/train_32x32.mat')
    test_data = sio.loadmat('C:/Users/320060820/experiments/datasets/svhn/test_32x32.mat')

    (x_train, y_train), (x_test, y_test) = (train_data['X'], train_data['y']), (test_data['X'], test_data['y'])

    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    x_train = x_train[np.newaxis,...]
    x_train = np.swapaxes(x_train,0,4).squeeze()

    x_test = x_test[np.newaxis,...]
    x_test = np.swapaxes(x_test,0,4).squeeze()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)