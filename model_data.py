import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K


def get_model_weights(model):
    """Get weights and biases of the CNN model. Includes Conv and Dense layers

    Returns:
        ndarray: concatenated weights
        ndarray: concatenated biases
    """
    
    weights_list = []
    biases_list = []
    for layer in model.layers:
        if layer.weights and 'kernel' in layer.weights[0].name:
            w = layer.get_weights()[0].flatten()
            b = layer.get_weights()[1].flatten()
            weights_list.append(w)
            biases_list.append(b)

    weights_arr = np.concatenate(weights_list)
    biases_arr = np.concatenate(biases_list)

    return weights_arr, biases_arr


def get_model_weights_by_layer(model, dense=False):
    """Get weights and biases of CNN model as a list, separately for each layer
        
    Args:
        dense (bool, optional): Include dense layer weights and biases. Defaults to False.
    Returns:
        list of ndarrays: weights and biases for each layer.
    """

    layer_weights = []

    layer_indices = get_conv_layer_indices(model)

    for i in layer_indices:
        layer_weights.append(model.layers[i].get_weights())
    
    if dense:
        layer_indices = get_dense_layer_indices(model)
        for i in layer_indices:
            layer_weights.append(model.layers)
    
    return layer_weights


def get_conv_layer_indices(model):
    """Get all indices of Convolutional layers in the Keras Model
    
    Args:
        model (Object): Keras Model
    
    Returns:
        list: conv layer indices
    """

    layer_num = []
    for i in range(len(model.layers)):
        if model.layers[i].name.startswith('conv'):
            layer_num.append(i)

    return layer_num


def get_dense_layer_indices(model):
    """Get all indices of Convolutional layers in the Keras Model
    
    Args:
        model (Object): Keras Model
    
    Returns:
        list: dense layer indices
    """
    layer_num = []
    for i in range(len(model.layers)):
        if model.layers[i].name.startswith('dense'):
            layer_num.append(i)

    return layer_num


def get_activation_maps(model, x_test_sample, layer_name):

    outputs = [layer.output for layer in model.layers if layer.name.startswith(layer_name)]
    activation_model = keras.models.Model(inputs=model.input, outputs=outputs)
    activations = activation_model.predict(x_test_sample)

    return activations

def get_num_params_per_layer(model, layer_names):
    """Get the number of parameters (weights and biases) for each layer
    
    Args:
        layer_names (sequence): Sequence of layer names
    
    Returns:
        sequence: Number of weights per layer
        sequence: Number of biases per layer
    """

    num_weights = []
    num_biases = []
    for layer in model.layers:
        if layer.name in layer_names:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            num_weights.append(len(w.flatten()))
            num_biases.append(len(b.flatten()))
    
    return num_weights, num_biases

def get_num_activations_per_layer(model, layer_names, exclude_kernels=False):
    """Get the number of activations for each layer
    
    Args:
        layer_names (sequence): Sequence of layer names
    
    Returns:
        sequence: Number of activations per layer
    """

    num_activations = []
    for layer in model.layers:
        if layer.name in layer_names:
            if exclude_kernels:
                num_activations.append(np.product(layer.output_shape[1:-1]))
            else:
                num_activations.append(np.product(layer.output_shape[1:]))
    
    return num_activations


class Model:
    """Class for containing properties of the Keras model
        
    Args:
        name (string): name of the model
        test_data (Tuple of ndarrays): Test data (x, y) to evaluate the network on
        model (Object, optional): Keras model object if loaded model already 
                                        available. Defaults to None.
        path (String, optional): Path to Keras model to load the Keras model from.
                                Defaults to None.
    """

    def __init__(self, name, test_data, model=None, path=None):


        self.name = name
        self.test_data = test_data
        self.path = path
        if self.path is not None:
            K.clear_session()
            self.model = keras.models.load_model(self.path)
        else:
            self.model = model

    def __str__(self):
        return self.name

    @property
    def x_test(self):
        """Test data X
        """
        return self.test_data[0]
    
    @property
    def y_test(self):
        """Test data y
        """
        return self.test_data[1]
    
    @property
    def conv_layer_indices(self):
        """Indices of convolutional layers
        """
        return get_conv_layer_indices(self.model)
    
    @property
    def dense_layer_indices(self):
        """Indices of dense layers
        """
        return get_dense_layer_indices(self.model)

    def load_model_from_path(self):
        """load keras model from given path
        """
        if self.path is not None:
            K.clear_session()
            self.model = keras.models.load_model(self.path)
        else:
            raise ValueError('Path variable is empty')

    def evaluate_accuracy(self):
        """Evaluate inference accuracy of the network

        Returns:
            list: Loss and accuracy of the model for the given test data
        """

        return self.model.evaluate(self.x_test, self.y_test, verbose = 0)
    
    def get_num_params_per_layer(self, layer_names):
        """Get the number of parameters (weights and biases) for each layer
        
        Args:
            layer_names (sequence): Sequence of layer names
        
        Returns:
            sequence: Number of weights per layer
            sequence: Number of biases per layer
        """

        num_weights = []
        num_biases = []
        for layer in self.model.layers:
            if layer.name in layer_names:
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                num_weights.append(len(w.flatten()))
                num_biases.append(len(b.flatten()))
        
        return num_weights, num_biases
    
    def get_num_activations_per_layer(self, layer_names):
        """Get the number of activations for each layer
        
        Args:
            layer_names (sequence): Sequence of layer names
        
        Returns:
            sequence: Number of activations per layer
        """

        num_activations = []
        for layer in self.model.layers:
            if layer.name in layer_names:
                num_activations.append(np.product(layer.output_shape[1:]))
        
        return num_activations


    def get_model_weights(self):
        """Get weights and biases of the CNN model. Includes Conv and Dense layers

        Returns:
            ndarray: concatenated weights
            ndarray: concatenated biases
        """
    
        weights_list = []
        biases_list = []
        for layer in self.model.layers:
            if layer.weights and 'kernel' in layer.weights[0].name:
                w = layer.get_weights()[0].flatten()
                b = layer.get_weights()[1].flatten()
                weights_list.append(w)
                biases_list.append(b)
        
        weights_arr = np.concatenate(weights_list)
        biases_arr = np.concatenate(biases_list)

        return weights_arr, biases_arr

    def get_model_weights_by_layer(self, dense=False):
        """Get weights and biases of CNN model as a list, separately for each layer
        
        Args:
            dense (bool, optional): Include dense layer weights and biases. Defaults to False.
        Returns:
            list of ndarrays: weights and biases for each layer.
        """
        layer_weights = []

        layer_indices = self.conv_layer_indices

        for i in layer_indices:
            layer_weights.append(self.model.layers[i].get_weights())
        
        if dense:
            layer_indices = self.dense_layer_indices
            for i in layer_indices:
                layer_weights.append(self.model.layers[i].get_weights())
        
        return layer_weights

    def get_activation_maps(self, x_test_sample, layer_name):
        """Get activation maps from a given layer after passing a test sample 
        image through the network
        
        Args:
            x_test_sample (ndarray): One or more images to pass through the network
            layer_name (string): Name of layer to collect values from
        
        Returns:
            list: List of activations from each kernel in the layer
        """
        outputs = [layer.output for layer in self.model.layers if layer.name == layer_name]
        activation_model = keras.models.Model(inputs=self.model.input, outputs=outputs)
        activations = activation_model.predict(x_test_sample)

        return activations
