import numpy as np
import keras
from keras.datasets import mnist
from skimage.transform import resize


def get_data(train = False):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    train_size = x_train.shape[0]
    test_size = x_test.shape[0]

    x_train_resize = np.zeros(shape = (train_size, 96, 96, 1))
    x_test_resize = np.zeros(shape = (test_size, 96, 96, 1))

    if train:
        for i in range(0, train_size):
            x_train_resize[i, :, :, :] = resize(x_train[i, :, :, :], (96, 96, 1))
    for i in range(0, test_size):
        x_test_resize[i, :, :, :]=resize(x_test[i, :, :, :], (96, 96, 1))

    x_train_resize = np.reshape(x_train_resize, (len(x_train_resize), 96, 96, 1))  
    x_test_resize = np.reshape(x_test_resize, (len(x_test_resize), 96, 96, 1))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train_resize, y_train), (x_test_resize, y_test)