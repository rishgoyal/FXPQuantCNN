import numpy as np
import model_data

def evaluate_memory_consumption(model, layer_names, quant_params, bitwidth=None):

    num_weights, num_biases = model_data.get_num_params_per_layer(model, layer_names)
    num_activations = model_data.get_num_activations_per_layer(model, layer_names)
    num_params = {'weights': num_weights, 'biases': num_biases, 'activations': num_activations}

    memory_consumption = {}
    for k in quant_params:
        bw = [quant_params[k][layer][0] for layer in quant_params[k]]
        memory_consumption[k] = sum([bw[i] * num_params[k][i] for i in range(len(layer_names))]) / (8*1024)
    
    return memory_consumption

def evaluate_multiplications_cost(model, layer_names, quant_params):

    num_weights, _ = model_data.get_num_params_per_layer(model, layer_names)
    num_activations = model_data.get_num_activations_per_layer(model, layer_names, exclude_kernels=True)

    cost = []

    for i in range(len(layer_names)):
        bw_w = quant_params['weights'][layer_names[i]][0]
        bw_a = quant_params['activations'][layer_names[i]][0]
        cost.append(np.uint64(bw_w * num_weights[i] * bw_a * num_activations[i]))
    
    return sum(cost)
