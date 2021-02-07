import numpy as np
seed = 42
np.random.seed(seed)


def model_size(model, per_line=False):
    """
    Prints details of the model

    Args:
        model: keras model

    Returns:
        number of floating point operations
    """
    X  = [(layer, [w.shape for w in layer.get_weights()], layer.input_shape[1:], layer.output_shape[1:]) for layer in model.layers]
    XW = [(layer, weights, input_shape, output_shape) for layer, weights, input_shape, output_shape in X if len(weights)>0]
    total = 0
    total=np.uint64(total)
    
    num_bytes = 4
    nb_neurons = 0
    for i, (layer, weights, input_shape, output_shape) in enumerate(XW):

        if len(weights) > 0: 
            nb_neurons += output_shape[-1]
        activation_mem = num_bytes * (np.product(input_shape) + np.product(output_shape))/1024**2
        if layer.__class__.__name__.startswith("Conv2D") or layer.__class__.__name__.startswith("atrousconv2d"):
            mul_adds = np.product(weights[0] + output_shape[:-1],dtype=np.uint64) # np.product(weights[0] + output_shape[1:])  for theano
            if per_line:
                print("{}, layer name = {:}, weights = {:}, input_shape = {:},  output_shape = {:}, # of flops = {:,}, memory: {:.2f} MB".format(i, layer.__class__.__name__, weights, 
                input_shape, output_shape, mul_adds, activation_mem ))
            total += mul_adds
        elif layer.__class__.__name__.startswith("separableconvolution"):
            mul_adds = np.product(weights[0] + output_shape[:-1]) + np.product(weights[1] + output_shape[:-1],dtype=np.uint64)
            if per_line:
                print("{}, layer name = {:}, weights = {:}, input_shape = {:},  output_shape = {:}, # of flops = {:,}, memory: {:.2f} MB".format(i, layer.__class__.__name__, weights, 
                input_shape, output_shape, mul_adds, activation_mem ))
            total += mul_adds

        elif layer.__class__.__name__.startswith("Dense") or layer.__class__.__name__.startswith("binary") or layer.__class__.__name__.startswith("nary"):
            mul_adds = np.product(weights[0] + output_shape[:-1],dtype=np.uint64)
            if per_line:
                print("{}, layer name = {:}, weights = {:}, input_shape = {:},  output_shape = {:}, # of flops = {:,}, memory: {:.2f} MB".format(i, layer.__class__.__name__, weights,
             input_shape, output_shape, mul_adds, activation_mem))
            total += mul_adds
        elif layer.__class__.__name__.startswith("Batch"):
            mul_adds = len(weights) * np.product(input_shape,dtype=np.uint64)
            if per_line:
                print("{}, layer name = {:}, weights = {:}, input_shape = {:},  output_shape = {:}, # of flops = {:,}, memory: {:.2f} MB".format(i, layer.__class__.__name__, weights, 
                input_shape, output_shape, mul_adds, activation_mem))
            total += mul_adds
        else:
            pass
#        print(" - Number of multiply-adds = {:,}".format(total))
        
    print(" - Number of multiply-adds = {:,}".format(total))

    num_params = model.count_params()
    print(" - Number of parameters: {:,} ".format(num_params))
#    print(" - Size in megabyte: {:.2f} MB".format(num_params*2/1024**2))

    num_layers = sum([1 for l in model.layers if l.__class__.__name__.startswith("Conv2D") or l.name.startswith("separableconvolution") or l.name.startswith("atrousconvolution")])
    print(" - Number of convolution layers: {}".format(num_layers))
    print(" - Number of neurons: {}".format(nb_neurons))
