from keras.callbacks import TensorBoard
#from keras import optimizers
import keras

import tensorflow as tf
import numpy as np
#from scipy.misc import imresize
from skimage.transform import resize

def convert_data(training_data, test_data, num_classes, reduced_training_size_factor=10, input_channels=1):
    
    (x_train, y_train), (x_test, y_test) = training_data, test_data

    train_size = int(len(x_train) / reduced_training_size_factor)
    test_size = int(len(x_test) / reduced_training_size_factor)
    x_train = x_train[0:train_size].astype('float32') / 255.
    x_test = x_test[0:test_size].astype('float32') / 255.
    y_train = y_train[0:train_size]
    y_test = y_test[0:test_size]

    # convert input data
    x_train_resize = np.zeros(shape = (train_size, 96, 96, input_channels))
    x_test_resize = np.zeros(shape = (test_size, 96, 96, input_channels))

    for i in range(0, train_size):
        x_train_resize[i, :, :, :] = resize(x_train[i, :, :, :], (96, 96, input_channels))
    for i in range(0, test_size):
        x_test_resize[i, :, :, :]=resize(x_test[i, :, :, :], (96, 96, input_channels))


    x_train_resize = np.reshape(x_train_resize, (len(x_train_resize), 96, 96, input_channels))  
    x_test_resize = np.reshape(x_test_resize, (len(x_test_resize), 96, 96, input_channels))

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train_resize, y_train), (x_test_resize, y_test)



def train_model(model, training_data, test_data, epochs, batch_size, num_classes, reduced_training_size_factor=10, rgb=False, cb=None, lr=0.01):
    """
    Converts the mnist data and trains the model. 
    rgb: Flag for type of input image. True if input image is RGB (3 channel)
    cb: list of Keras callbacks during training
    """
    if rgb:
        input_channels = 3
    else:
        input_channels = 1
    # training_data, test_data = convert_data(training_data, test_data, batch_size, num_classes, reduced_training_size_factor, input_channels)

    (x_train_resize, y_train), (x_test_resize, y_test) = training_data, test_data
    # Compile, save and train the model
    model.compile(loss=keras.losses.categorical_crossentropy,
#        optimizer=keras.optimizers.Adadelta(),
        # optimizer=keras.optimizers.rmsprop(lr=0.0001),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=['accuracy'])

    # model.save('inception_v3_model_untrained.h5')

    call_backs = [TensorBoard(log_dir='/tmp/keras_cnn/run1')]
    if cb is not None:
        call_backs.extend(cb)
    history = model.fit(x_train_resize, y_train,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = (x_test_resize, y_test),
        callbacks = call_backs)

    base_score = model.evaluate(x_test_resize, y_test, verbose=0)
    print('Test loss:', base_score[0])
    print('Test accuracy:', base_score[1])

    # model.save('inception_v3_model_batchnorm.h5')
    # print('Trained model saved as inception_v3_model_batchnorm.h5')

    return (x_train_resize, y_train), (x_test_resize, y_test), model, history
    
