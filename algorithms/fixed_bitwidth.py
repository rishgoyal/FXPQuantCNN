import keras
import tensorflow as tf
import numpy as np

import fxp_quantize
import model_data


def calculate_fractional_offset(input_arr, bitwidth=8):
    """Calculate the fractional offset for the fixed-point representation
    with a bitwidth of 8 given an input array such that no values are clipped.
    
    Args:
        input_arr (ndarray): Input array of values
        bitwidth (int): Bitwidth of the targeted fixed-point representation.
                        Defaults to 8.
    
    Returns:
        int: Calculated fractional offset
    """
    
    if bitwidth < 2:
        raise ValueError('Bitwidth must be larger than 1.')

    sign_bit = 1
    largest_value = np.max(np.abs(input_arr))
    int_bits = np.ceil(np.log2(largest_value))
    fractional_offset = bitwidth - sign_bit - int_bits
    return int(fractional_offset)


def find_offset_for_layer_weights(model_obj, layer_name, bitwidth, parameter_type):
    """Find fractional offsets for weights of a given layer for a specific bitwidth
    such that clipping is avoided
    
    Args:
        model_obj (Object): model_data.Model object
        layer_name (str): Name of layer
        bitwidth (int): Bitwidth of the fixed-point representation
        parameter_type (int): Find offsets for weights (0) or biases (1)
    
    Returns:
        int: Fractional offset for that layer
    """
    if not (parameter_type == 0 or parameter_type == 1):
        raise ValueError('Parameter type must be 0 (Weights) or 1 (Biases)')

    for layer in model_obj.model.layers:
        if layer.weights and 'kernel' in layer.weights[0].name and layer_name == layer.name:
            return calculate_fractional_offset(layer.get_weights()[parameter_type], bitwidth)


def find_offsets_for_model_weights(model_obj, parameter_type, bitwidth):
    """Find the fractional offsets for a given model's weights or biases
    for each layer in the model
    
    Args:
        model_obj (Object): model_data.Model object
        parameter_type (int): Find offsets for weights (0) or biases (1)
        bitwidth (int): Bitwidth of the targeted fixed-point representation
    Returns:
        int: Calculated fractional offset
    """
    
    fractional_offsets = {}

    if not (parameter_type == 0 or parameter_type == 1):
        raise ValueError('Parameter type must be 0 (Weights) or 1 (Biases)')
    
    for layer in model_obj.model.layers:
        if layer.weights and 'kernel' in layer.weights[0].name:
            fractional_offsets[layer.name] = calculate_fractional_offset(layer.get_weights()[parameter_type], bitwidth)

    return fractional_offsets


def evaluate_quantized_weights(model_obj, original_acc, bitwidth=8):
    """Evaluate inference of quantized network using fixed-bitwidth and no clipping
    for weights
    
    Args:
        model_obj (Object): model_data.Model object (the object must have the path stored!)
        bitwidth (int): Bitwidth of the targeted fixed-point representation.
                        Defaults to 8.

    Returns:
        dict: Parameters for the quantized weights
        dict: Parameters for the quantized biases
        float: Fraction of accuracy drop
    """
    
    print('Calculating fractional offsets for weights of the network')
    # Calculate offsets and create dictionary for quantization parameters
    offsets = find_offsets_for_model_weights(model_obj, 0, bitwidth)
    w_quant_params = {i: [bitwidth, j] for i, j in offsets.items()}

    # Evaluate original network's accuracy
    print(f'Original network\'s accuracy: {original_acc}')

    model_obj = fxp_quantize.fix_weights_quantization(model_obj, w_quant_params)
    
    # Evaluate quantized network's accuracy
    new_acc = model_obj.evaluate_accuracy()[1]
    print(f'Quantized network\'s accuracy: {new_acc}')
    print(f'Accuracy drop: {((original_acc - new_acc)/original_acc * 100):.3f} %')
    acc_drop = {'weights': ((original_acc - new_acc)/original_acc * 100)}
    print('Calculating fractional offsets for the biases of the network')

    # Calculate offsets and create dictionary for quantization parameters
    offsets = find_offsets_for_model_weights(model_obj, 1, bitwidth)
    b_quant_params = {i: [bitwidth, j] for i, j in offsets.items()}

    model_obj = fxp_quantize.fix_biases_quantization(model_obj, b_quant_params)
    
    # Evaluate quantized network's accuracy
    new_acc = model_obj.evaluate_accuracy()[1]
    print(f'Quantized network\'s accuracy: {new_acc}')
    print(f'Accuracy drop: {((original_acc - new_acc)/original_acc * 100):.3f} %')
    acc_drop['biases'] = ((original_acc - new_acc)/original_acc * 100)

    return model_obj, w_quant_params, b_quant_params, acc_drop


def find_offset_for_layer_activations(model_obj, layer_name, bitwidth, num_images=250):
    """Find the fractional offsets for a given model's activations for a specified layer
    
    Args:
        model_obj (Object): model_data.Model object
        layer_name (str): Name of the layer
        bitwidth (int): Bitwidth of the target representation
        num_images (int, optional): Number of images to use for generating 
        distribution of activations. Defaults to 250.
    
    Returns:
        int: Fractional offset for the layer
    """

    for layer in model_obj.model.layers:
        if (layer.name.startswith('conv2d') or layer.name.startswith('dense') or layer.name.startswith('binary')) and (layer_name == layer.name):
            act_maps = model_obj.get_activation_maps(model_obj.x_test[:num_images], layer.name)
            return calculate_fractional_offset(act_maps, bitwidth)


def find_offsets_for_model_activations(model_arch, name, test_data, bitwidth, num_images=250):
    """Find the fractional offsets for a given model's activations for each
    layer in the model
    
    Args:
        model_arch (Object): Object from model_gen
        name (string): Name of the model
        test_data (sequence): Test data needed for inference. Sequence of ndarrays
        bitwidth (int): Bitwidth of the targeted fixed-point representation
        num_images (int): Number of images to collect activation maps for.
                            Defaults to 10.
    
    Returns:
        dict: Calculated fractional offsets
    """

    fractional_offsets = {}
    model_obj = model_data.Model(name, test_data, model=model_arch.get_float_model())

    for layer in model_obj.model.layers:
        if layer.name.startswith('conv2d') or layer.name.startswith('dense') or layer.name.startswith('binary'):
            act_maps = model_obj.get_activation_maps(model_obj.x_test[:num_images], layer.name)
            fractional_offsets[layer.name] = calculate_fractional_offset(act_maps, bitwidth)

    return fractional_offsets


def evaluate_quantized_activations(model_arch, name, test_data, original_acc, bitwidth=8, num_images=250):
    """Evaluate inference of quantized network using fixed-bitwidth and no clipping
    for weights
    
    Args:
        model_arch (Object): Object from model_gen
        name (string): Name of the model
        test_data (sequence): Test data needed for inference. Sequence of ndarrays
                                (x_test, y_test)
        bitwidth (int): Bitwidth of the targeted fixed-point representation.
                        Defaults to 8.
        num_images (int): number of images to collect activation maps for. 
                            Defaults to 10
    
    Returns:
        dict: Parameters for quantized activations
    """
    
    offsets = find_offsets_for_model_activations(model_arch, name, test_data, bitwidth, num_images)
    act_quant_params = {i: [bitwidth, j] for i, j in offsets.items()}
    
    print(f'Original network\'s accuracy: {original_acc}')
    
    quant_model = model_data.Model(name, test_data, model=model_arch.get_fxp_model(act_quant_params))
    new_acc = quant_model.evaluate_accuracy()[1]
    print(f'Quantized network\'s accuracy: {new_acc}')
    print(f'Accuracy drop: {((original_acc - new_acc)/original_acc * 100):.3f} %')
    acc_drop = {'activations': ((original_acc - new_acc)/original_acc * 100)}
    
    return quant_model, act_quant_params, acc_drop
