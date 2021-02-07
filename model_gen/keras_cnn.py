from keras import models, layers, optimizers
import numpy as np
import tensorflow as tf

from convert_float_fixed import ConvertFloatFixed

from keras import backend as K
K.clear_session()



class KerasCNN:
    """Class for the simple sequential Keras CNN with lambda layers (for fixed point representation)
    or as original floating point model

    Args:
    input_shape (Tuple of integers): Input shape for input layer
    num_outputs (integer): number of output neurons / number of classes
    path_trained_weights (string, optional): Absolute or relative path to weights if 
                                            weights must be loaded when loading the 
                                            model. Defaults to None.
    """

    def __init__(self, input_shape, num_outputs, path_trained_weights=None):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.path_trained_weights = path_trained_weights
    
    @staticmethod
    def _add_conv_layer(model, filters, kernel_size, activation, quant_params={}, 
                    input_shape=None, padding='valid'):
        """Add convolution layer with Lambda or BN layers
        
        Args:
            model (Object): Keras model object
            filters (integer): Number of convolutional kernels
            kernel_size (2-Tuple of integers): Size of the kernel
            activation (string): activation of the convolutional layer
            quant_params (dict, optional): layer names as keys with values of [bw, f] 
                                            for lambda layer if quantization layer must 
                                            be added. Defaults to {}.
            input_shape (Tuple of integers, optional): Input shape of convolutional layer. 
                                                        Only needed for input layer.
                                                        Defaults to None.
            padding (str, optional): Padding for convolutional layer. Defaults to 'valid'.
        """
    
        if input_shape is not None:
            model.add(layers.Conv2D(filters, kernel_size, activation=activation, padding=padding, 
                        input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, kernel_size, activation=activation, padding=padding))

        if model.layers[-1].name in quant_params.keys():
            k = model.layers[-1].name
            cff = ConvertFloatFixed(quant_params[k][0], quant_params[k][1])
            model.add(layers.Lambda(cff.quantize_tf))

        model.add(layers.BatchNormalization())

        return model

    def get_fxp_model(self, quant_params):
        """Get CNN model with lambda layers for quantizing layer outputs/activations
        of the Keras Vanilla CNN structure
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to

        Returns:
            Object: Compiled Keras model
        """

        K.clear_session()

        model = models.Sequential()
        model = self._add_conv_layer(model, 32, (3, 3), 'relu',
                                    quant_params=quant_params,
                                    input_shape=self.input_shape)
        model = self._add_conv_layer(model, 32, (3, 3), 'relu',
                                    quant_params = quant_params)
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model = self._add_conv_layer(model, 64, (3, 3), 'relu', 
                                    quant_params=quant_params,
                                    padding='same')
        model = self._add_conv_layer(model, 64, (3, 3), 'relu',
                                    quant_params=quant_params)
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.GlobalMaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(self.num_outputs))

        for k in quant_params.keys():
            if model.layers[-1].name == k:
                cff = ConvertFloatFixed(quant_params[k][0], quant_params[k][1])
                model.add(layers.Lambda(cff.quantize_tf))
                break
        
        model.add(layers.Activation('softmax'))

        # copy weights to model
        model.load_weights(self.path_trained_weights)

        opt = optimizers.rmsprop(lr=0.0001)

        model.compile(loss='categorical_crossentropy', 
                        optimizer = opt,
                        metrics=['accuracy'])

        return model

    def get_float_model(self):
        """Get floating point precision model of the Keras vanilla CNN
        
        Returns:
            Object: Compiled Keras model
        """

        K.clear_session()
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.GlobalMaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(self.num_outputs))
        model.add(layers.Activation('softmax'))

        # copy weights to model
        if self.path_trained_weights:
            model.load_weights(self.path_trained_weights)

        opt = optimizers.rmsprop(lr=0.0001)

        model.compile(loss='categorical_crossentropy', 
                        optimizer = opt,
                        metrics=['accuracy'])

        return model


class KerasCNNLarge:
    """Class for a longer/larger sequential Keras CNN with lambda layers 
    (for fixed point representation) or as original floating point model
    
    Args:
    input_shape (Tuple of integers): Input shape for input layer
    num_outputs (int): number of output neurons / number of classes
    num_kernels (int): number of kernels for the first half of the network
    num_stages (int): number of stages to add. Each stage consists of 2 convolutional layers. Defaults to 2.
    pool_layer_interval (int): stage intervals at which to place the max pooling layer. Defaults to 1
    path_trained_weights (string, optional): Absolute or relative path to weights if 
                                            weights must be loaded when loading the 
                                            model. Defaults to None.
    """


    def __init__(self, input_shape, num_outputs, num_kernels, num_stages=2, pool_layer_interval=1, path_trained_weights=None):
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.num_kernels = num_kernels
        self.num_stages = num_stages
        self.pool_layer_interval = pool_layer_interval
        self.path_trained_weights = path_trained_weights
    

    @staticmethod
    def _add_conv_layer(model, num_kernels, kernel_size=(3, 3), activation='relu', quant_params={}, 
                    input_shape=None, padding='same'):


        if input_shape is not None:
            model.add(layers.Conv2D(num_kernels, kernel_size, activation=activation, input_shape=input_shape))
        else:
            model.add(layers.Conv2D(num_kernels, kernel_size, activation=activation, padding=padding))

        if model.layers[-1].name in quant_params.keys():
            k = model.layers[-1].name
            cff = ConvertFloatFixed(quant_params[k][0], quant_params[k][1])
            model.add(layers.Lambda(cff.quantize_tf))

        model.add(layers.BatchNormalization())

        return model

    def get_fxp_model(self, quant_params):
        """Get CNN model with lambda layers for quantizing layer outputs/activations
        of the Keras Vanilla CNN structure
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to
        
        Returns:
            Object: Compiled Keras model
        """
        
        K.clear_session()
        model = models.Sequential()
        kernels = self.num_kernels
        for i in range(self.num_stages):
            if i == 0:
                model = self._add_conv_layer(model, kernels, (3, 3), input_shape = self.input_shape, 
                                            quant_params=quant_params)
            else:
                model = self._add_conv_layer(model, kernels, (3, 3), quant_params=quant_params)
            
            model = self._add_conv_layer(model, kernels, (3, 3), quant_params=quant_params)

            if (i + 1) % self.pool_layer_interval == 0:
                model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
                model.add(layers.Dropout(0.25))

            if (i + 1) * 2 == np.floor(self.num_stages / 2. ) * 2:
                kernels *= 2
        
        model.add(layers.GlobalMaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(self.num_outputs))
        if model.layers[-1].name in quant_params.keys():
            k = model.layers[-1].name
            cff = ConvertFloatFixed(quant_params[k][0], quant_params[k][1])
            model.add(layers.Lambda(cff.quantize_tf))
        model.add(layers.Activation('softmax'))

        opt = optimizers.rmsprop(lr=0.0001)
        
        # copy weights to model
        if self.path_trained_weights:
            model.load_weights(self.path_trained_weights)

        model.compile(loss='categorical_crossentropy', 
                        optimizer = opt,
                        metrics=['accuracy'])

        return model

    def get_float_model(self):
        """Get floating point precision model of the Keras vanilla CNN
        
        Returns:
            Object: Compiled Keras model
        """

        K.clear_session()

        model = models.Sequential()
        kernels = self.num_kernels
        for i in range(self.num_stages):
            if i == 0:
                model.add(layers.Conv2D(kernels, (3, 3), activation='relu', input_shape=self.input_shape))
            else:
                model.add(layers.Conv2D(kernels, (3, 3), padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(kernels, (3, 3), padding='same', activation='relu'))
            model.add(layers.BatchNormalization())

            if (i + 1) % self.pool_layer_interval == 0:
                model.add(layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
                model.add(layers.Dropout(0.25))

            if (i + 1) * 2 == np.floor(self.num_stages / 2. ) * 2:
                kernels *= 2
        
        model.add(layers.GlobalMaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(self.num_outputs))
        model.add(layers.Activation('softmax'))

        # copy weights to model
        if self.path_trained_weights:
            model.load_weights(self.path_trained_weights)

        opt = optimizers.rmsprop(lr=0.0001)

        model.compile(loss='categorical_crossentropy', 
                        optimizer = opt,
                        metrics=['accuracy'])

        return model
