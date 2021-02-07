import tensorflow as tf
import numpy as np

from convert_float_fixed import ConvertFloatFixed
import model_data


def fix_model_quantization(model_arch, model_name, test_data, quant_params):
    """Fix quantization parameters for the entire model (Weights, biases and activations)
    
    Args:
        model_arch (Object): Model architecture - Keras model built from functions in model_gen
        model_name (str): Name of the model
        test_data (tuple): Test data (x, y) for evaluation
        quant_params (namedtuple): Namedtuple QuantizationParameters (found in optimized_search.py)
    
    Returns:
        Object: model_data.Model object
    """
    
    if quant_params.activations:
        model = model_arch.get_fxp_model(quant_params.activations)
    else:
        model = model_arch.get_float_model()

    model_obj = model_data.Model(model_name, test_data, model=model)

    if quant_params.weights:
        model_obj = fix_weights_quantization(model_obj, quant_params.weights)
    if quant_params.biases:
        model_obj = fix_biases_quantization(model_obj, quant_params.biases)

    return model_obj


def fix_weights_quantization(model_obj, parameters):
    """Fix the quantization parameters (BW, F) for the weights of 
    the given layers of the given model
    
    Args:
        model_obj (Object): model_data.Model object
        parameters (dict): Parameters for the layers for which 
                            quantization needs to be fixed. 
                            {'name': [bw, f]}

    Returns:
        Object: model_data.Model object
    """
    
    for k in parameters.keys():
        model_obj.model = quantize_weights(model_obj.model, parameters[k][0], parameters[k][1],
                                                   layer_name=[k])
    return model_obj


def fix_biases_quantization(model_obj, parameters):
    """Fix the quantization parameters (BW, F) for the biases of 
    the given layers of the given model
    
    Args:
        model_obj (Object): model_data.Model object
        parameters (dict): Parameters for the layers for which 
                            quantization needs to be fixed. 
                            {'name': [bw, f]}

    Returns:
        Object: model_data.Model object
    """
    
    for k in parameters.keys():
        model_obj.model = quantize_biases(model_obj.model, parameters[k][0], parameters[k][1],
                                                   layer_name=[k])
    return model_obj


def quantize_weights(model, bitwidth, fractional_bits, layer_name='all', biases=False, sqrt=False):
    """Quantize weights of a given set of layers of a given CNN model to a specified 
    bitwidth and fractional offset
    
    Args:
        model (Object): Keras model Object
        bitwidth (integer): Bitwidth for target representation
        fractional_bits (integer): Fractional offset for target representation
        layer_name (list, optional): List of layer names to quantize as taken from Keras model.
                                    Defaults to 'all'.
        biases (bool, optional): True if biases are to be quantized. Defaults to False. Defaults to False
        sqrt (bool, optional): True if non-uniform square root quantization must be applied. Defaults to False
    """

    if layer_name == 'all':
        layers = model.layers[:]
    else:
        layers = []
        for n in layer_name:
            for layer in model.layers:
                if layer.name == n:
                    layers.append(layer)
                    break

    quantizer = ConvertFloatFixed(bitwidth, fractional_bits)

    for layer in layers:
        if layer.weights and 'kernel' in layer.weights[0].name:
            layer_weights = layer.get_weights()

            if sqrt:
                layer_weights[0] = quantizer.quantize_sqrt(layer_weights[0])
            else:
                layer_weights[0] = quantizer.quantize(layer_weights[0])
            if biases:
                layer_weights[1] = quantizer.quantize(layer_weights[1])
            
            layer.set_weights(layer_weights)
    
    return model


def quantize_biases(model, bitwidth, fractional_bits, layer_name='all'):
    """Quantize biases of a given set of layers of a given CNN model to a specified 
    bitwidth and fractional offset
    
    Args:
        model (Object): Keras model Object
        bitwidth (integer): Bitwidth for target representation
        fractional_bits (integer): Fractional offset for target representation
        layer_name (str, optional): List of layer names to quantize as taken from Keras model.
                                    Defaults to 'all'.
    """
    if layer_name == 'all':
        layers = model.layers[:]
    else:
        layers = []
        for n in layer_name:
            for layer in model.layers:
                if layer.name == n:
                    layers.append(layer)
                    break

    quantizer = ConvertFloatFixed(bitwidth, fractional_bits)

    for layer in layers:
        if layer.weights and 'kernel' in layer.weights[0].name:
            layer_weights = layer.get_weights()
            layer_weights[1] = quantizer.quantize(layer_weights[1])
            layer.set_weights(layer_weights)
    
    return model
