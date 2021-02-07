 # -*- coding: utf-8 -*-
"""Inception V3 model fixed point

# Reference:

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

Code adapted from: https://raw.githubusercontent.com/fchollet/keras/master/keras/applications/inception_v3.py

# generates json model and png to visualise the model

"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import warnings

from keras.models import Model
from keras.layers import Activation, Flatten, Dense, Input, BatchNormalization, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Cropping2D
from keras.layers import Lambda, GlobalAveragePooling2D, Activation, GlobalMaxPooling1D

from keras.layers.core import Reshape

from keras import backend as K
from keras.utils.vis_utils import plot_model#SN

# from dragonfly.models.model_size import model_size
import tensorflow as tf
# to restart layer numbering
K.clear_session()#SN

BATCHNORMALIZATION = True
#BATCHNORMALIZATION = False


def fxp_quant(X, fractional_bits):
    
	A = K.maximum(0.0, X)
	int_val = A * (2 ** fractional_bits)
	integer_value = K.round(int_val)
	B = integer_value / (2 ** fractional_bits)
    
	return B

def conv2d_bn_fxp(x, nb_filter, nb_row, nb_col, fractional_bits, padding='same', subsample=(1, 1)):
	"""
	Utility function to apply conv + BN
	
	Args:
		x: keras tensor
		nb_filter: number of filters
		... 
	
	Returns:
		keras tensor
	"""
	x = Conv2D(nb_filter, (nb_row, nb_col), strides=subsample, activation='relu', padding=padding)(x)

	x = Lambda(fxp_quant, arguments={'fractional_bits': fractional_bits})(x)

	if BATCHNORMALIZATION:
		x = BatchNormalization()(x)
	return x


def get_model(input_shape, nb_classes, nb_fabrics, scale_factor, resampling, frac_bits=8, field_of_views='max', weights_file=None):
	"""
	Create inception_v3

	Args:
		inputs_shape: input image shape 
		nb_classes: number of classes (tough, delicate)
		nb_fabrics: number of fabric types
		scale_factor: integer from 2 to 5
		resampling: 'avg' or 'conv'
		field_of_views:
		weights_file

	Returns: 
		A Keras model instance

	
	input_image = Input(shape=input_shape)

	supported_preprocessings = ['avg', 'conv']
	assert(resampling in supported_preprocessings), "resampling should be one of {}".format(supported_preprocessings)

	if field_of_views == 'all':
		x = combine_all_fovs(input_image, resampling=resampling)
	else:
		x = resample_image(input_image, scale_factor=scale_factor, method=resampling)
	"""
	input_image = Input(shape=input_shape)
	x = conv2d_bn_fxp(input_image, 20, 3, 3, frac_bits, subsample=(2,2))
	# x = conv2d_bn_fxp(input_image, 10, 3, 3, subsample=(2,2)) #half network depth
	
	strides=(2,2)
	nb_filter  = 16
	# nb_filter  = 8 #half network depth
	nb_inception_modules = 3
	for _ in range(nb_inception_modules):
		branch1x1 = conv2d_bn_fxp(x, nb_filter, 1, 1, frac_bits, subsample=strides)

		branch3x3sgl = conv2d_bn_fxp(x, nb_filter, 1, 1, frac_bits)
		branch3x3sgl = conv2d_bn_fxp(branch3x3sgl, nb_filter, 3, 3, frac_bits, subsample=strides)

		branch3x3dbl = conv2d_bn_fxp(x, nb_filter, 1, 1, frac_bits)
		branch3x3dbl = conv2d_bn_fxp(branch3x3dbl, nb_filter, 3, 3, frac_bits)
		branch3x3dbl = conv2d_bn_fxp(branch3x3dbl, nb_filter, 3, 3, frac_bits, subsample=strides)

		branch_pool = AveragePooling2D((2, 2), strides=strides, padding='same')(x)
		
		branch_pool = conv2d_bn_fxp(branch_pool, nb_filter, 1, 1, frac_bits)
		x = concatenate([branch1x1, branch3x3sgl, branch3x3dbl, branch_pool], axis=-1) #	channel_axis = -1 # tensorflow
		nb_filter *=2

	
	x = GlobalAveragePooling2D()(x)
    #replace GlobalAveragePooling2D by AveragePooling2d with kernel size input size (for matlab codegen)
#	x = AveragePooling2D((6, 6), padding='valid')(x)
#	x = Flatten()(x)

    
	#output_N = Dense(nb_fabrics, activation='softmax', name='nary')(x)
	#output_B = Dense(nb_classes, activation='softmax', name='binary')(output_N)
	output_B = Dense(nb_classes, activation='softmax', name='binary')(x)
	#output_B = Lambda(noisy_or, output_shape=noisy_or_output_shape, name='binary')(output_N)
	
	#model = Model(input_image, [output_B, output_N])
	model = Model(input_image, [output_B])
	
	if weights_file is not None:
		model.load_weights(weights_file, by_name=True)

	return model

def save_model(filename, model):
	""" 
	Saves model

	Args:
	filename: name of model
	model: keras model

	Args:
	None
	"""
	# serialize model to JSON
	model_json = model.to_json()
	with open(filename, "w") as json_file:
		json_file.write(model_json)


def main():
	#res = 96*5
	res = 96
	f = 8
	model = get_model(input_shape=(res, res, 1), nb_classes=10, nb_fabrics=49, scale_factor=res//96, resampling='conv', frac_bits=f)
	print("model summary: ", model.summary())
	# model_size(model)
	fpath='Activations/inception_v3_fxp_model_batchnorm.json'
	save_model(fpath, model)
	plot_model(model, to_file='Activations/inception_v3_fxp_model_batchnorm.png', show_shapes=True, show_layer_names=True)#SN
    

if __name__ == "__main__":
	main()
				  
