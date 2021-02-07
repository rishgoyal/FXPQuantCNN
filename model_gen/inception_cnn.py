import numpy as np
import warnings

from keras import optimizers
from keras.models import Model
from keras.layers import Activation, Flatten, Dense, Input, BatchNormalization, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.layers import Lambda, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.layers.core import Reshape

from convert_float_fixed import ConvertFloatFixed

from keras import backend as K
from keras.utils.vis_utils import plot_model#SN

import tensorflow as tf
K.clear_session()#SN



def _add_conv_layer(x, filters, kernel_size, padding='same', subsample=(1, 1),
                quant_params={}):

    x = Conv2D(filters, kernel_size, strides=subsample, activation='relu', padding=padding)(x)

    # Add quantization layer if parameters are given
    layer = x.name.split('/')[0]
    if layer in quant_params.keys():
        cff = ConvertFloatFixed(quant_params[layer][0], quant_params[layer][1])
        x = Lambda(cff.quantize_tf)(x)
    x = BatchNormalization()(x)
    
    return x


class InceptionCNN:
    """Class for the Inception CNN as a fixed-point model with lambda layers 
    (for fixed point representation) or as original floating point model

    Args:
        input_shape (Tuple of integers): Input shape for input layer
        num_outputs (integer): number of output neurons / number of classes
        path_trained_weights (string, optional): Absolute or relative path to weights if 
                                                weights must be loaded when loading the 
                                                model. Defaults to None.
        dropout (bool): True if dropout should be added
        pool_layer_type (string): Max or average pooling
    """

    def __init__(self, input_shape, num_outputs, 
                dropout, pool_layer_type, path_trained_weights=None):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.path_trained_weights = path_trained_weights
        self.pool_layer_type = pool_layer_type
        self.dropout = dropout

    def get_fxp_model(self, quant_params):
        """Get CNN model with lambda layers for quantizing layer outputs/activations
        
        Args:
            quant_params (dict): Dictionary of layer names as keys with values of
                                [bw, f] to quantize the layer's output to
        """
        K.clear_session()
        
        input_image = Input(shape=self.input_shape)
        x = _add_conv_layer(input_image, 20, (3, 3), subsample=(2, 2), quant_params=quant_params)

        strides = (2, 2)
        filters = 16
        num_inception_layers = 3

        for _ in range(num_inception_layers):

            branch1x1 = _add_conv_layer(x, filters, (1, 1), subsample=strides,
                                        quant_params=quant_params)

            branch3x3sgl = _add_conv_layer(x, filters, (1, 1), 
                                            quant_params=quant_params)
            branch3x3sgl = _add_conv_layer(branch3x3sgl, filters, (3, 3), subsample=strides, 
                                            quant_params=quant_params)

            branch3x3dbl = _add_conv_layer(x, filters, (1, 1),
                                            quant_params=quant_params)
            branch3x3dbl = _add_conv_layer(branch3x3dbl, filters, (3, 3), 
                                            quant_params=quant_params)
            branch3x3dbl = _add_conv_layer(branch3x3dbl, filters, (3, 3), subsample=strides, 
                                            quant_params=quant_params)

            if self.pool_layer_type == 'average':
                branch_pool = AveragePooling2D((2, 2), strides=strides, padding='same')(x)
            elif self.pool_layer_type == 'max':
                branch_pool = MaxPooling2D((2, 2), strides=strides, padding='same')(x)

            if self.dropout:
                branch_pool = Dropout(0.5)(branch_pool)

            branch_pool = _add_conv_layer(branch_pool, filters, (1, 1), quant_params=quant_params)
            x = concatenate([branch1x1, branch3x3sgl, branch3x3dbl, branch_pool], axis=-1)

            filters *= 2
        
        if self.pool_layer_type == 'average':
            x = GlobalAveragePooling2D()(x)
        elif self.pool_layer_type == 'max':
            x = GlobalMaxPooling2D()(x)

        if self.dropout:
            x = Dropout(0.5)(x)
        
        x = Dense(self.num_outputs, name='binary')(x)

        layer = x.name.split('/')[0]
        if layer in quant_params.keys():
            cff = ConvertFloatFixed(quant_params[layer][0], quant_params[layer][1])
            x = Lambda(cff.quantize_tf)(x)
    
        output_B = Activation('softmax')(x)

        model = Model(input_image, [output_B])
        if self.path_trained_weights:
            model.load_weights(self.path_trained_weights, by_name=True)

        opt = optimizers.Adam(lr=0.01)

        model.compile(loss='categorical_crossentropy', 
                      optimizer=opt, 
                      metrics=['accuracy'])

        return model

    def get_float_model(self):
        """Get floating point precision model of the Keras vanilla CNN
        """
        K.clear_session()
        
        input_image = Input(shape=self.input_shape)
        x = _add_conv_layer(input_image, 20, (3, 3), subsample=(2, 2))

        strides = (2, 2)
        filters = 16
        num_inception_layers = 3

        for _ in range(num_inception_layers):

            branch1x1 = _add_conv_layer(x, filters, (1, 1), subsample=strides)

            branch3x3sgl = _add_conv_layer(x, filters, (1, 1))
            branch3x3sgl = _add_conv_layer(branch3x3sgl, filters, (3, 3), subsample=strides)

            branch3x3dbl = _add_conv_layer(x, filters, (1, 1))
            branch3x3dbl = _add_conv_layer(branch3x3dbl, filters, (3, 3))
            branch3x3dbl = _add_conv_layer(branch3x3dbl, filters, (3, 3), subsample=strides)

            if self.pool_layer_type == 'average':
                branch_pool = AveragePooling2D((2, 2), strides=strides, padding='same')(x)
            elif self.pool_layer_type == 'max':
                branch_pool = MaxPooling2D((2, 2), strides=strides, padding='same')(x)

            if self.dropout:
                branch_pool = Dropout(0.5)(branch_pool)

            branch_pool = _add_conv_layer(branch_pool, filters, (1, 1))
            x = concatenate([branch1x1, branch3x3sgl, branch3x3dbl, branch_pool], axis=-1)

            filters *= 2
        
        if self.pool_layer_type == 'average':
            x = GlobalAveragePooling2D()(x)
        elif self.pool_layer_type == 'max':
            x = GlobalMaxPooling2D()(x)

        if self.dropout:
            x = Dropout(0.5)(x)
        x = Dense(self.num_outputs, name='binary')(x)
        output_B = Activation('softmax')(x)

        model = Model(input_image, [output_B])
        if self.path_trained_weights:
            model.load_weights(self.path_trained_weights)

        opt = optimizers.Adam(lr=0.01)

        model.compile(loss='categorical_crossentropy', 
                      optimizer=opt, 
                      metrics=['accuracy'])

        return model

