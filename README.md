# Thesis Project on Convolutional Neural Network Quantization

Code used for the Master Thesis project on "Fixed-point Quantization of Convolutional Neural Networks for Quantized Inference on Embedded Platforms". Paper can be found at https://arxiv.org/abs/2102.02147


Path names are hard coded and so need to be altered before using this code. Usage of algorithms can be found in the jupyter notebooks under the analysis/ folder.

As for quantizing a CNN model, I first load an fxp_model from model_gen to quantize activations, then pass that model to functions in fxp_quantize.py to quantize weights and biases. The function evaluate() for class QuantizationEvaluator under optimized_search.py shows how this is performed.

Following is an overview of content in the respective directories

- algorithms/
    - brute_force.py - Brute force BW, F. Also found in jupyter notebooks in analysis/Brute Force Analysis/
    - fixed_bitwidth.py - Fixed 8-bit implementation
    - ind_optimized_search.py - Layer and parameter independent optimized search
    - optimized_search.py - Final working method of Optimized Search algorithm.
- analysis/ -  Main work on analyzing effects of quantization on various models for weights and activations
    - Brute Force Analysis/ - Notebooks on Brute Force Analysis for weights and activations
    - Final results (Linear Optimized Search)/ - Final tests comparing fixed bitwidth and optimized search
    - Fixed Bitwidth/ - Notebook to vary bitwidth approach
    - Layer Dependent Optimized Search/ - Parameter independent, Layer dependent optimized search
    - Layer Independent Optimized Search/ - Parameter and layer independent optimized search
    - Misc Experiments with Weights/
    - Weight Distributions/ - Weight distributions plotted for DF networks
    - Plotting.ipynb - Plotting misc results for report
- model_gen/ - scripts to generate the model structures in Keras. All classes have 2 functions. get_fxp_model() is used to load a model with a lambda layer to quantize activations. get_float_model() is used to load the original full precision model.
    - inception_cnn.py - Build the inception cnn model with pre-trained weights
    - keras_cnn.py - Two classes to build two different sequential CNN models with pre-trained weights
- model_archive/ - saved NN models
- model_training - Training of models on datasets
- outputs/ - data from experimentation. Code for storing and loading of the data can be found in the respective notebooks under analysis/
    - Brute_Force_Analysis/ - Brute force analysis results
    - Comp_Dependent_Optimized_Search - Final results after running optimized_search.py
    - Dependent_Optimized_Search - Layer-dependent but parameter independent quantization results
    - Independent_Optimized_Search - Layer and parameter independent optimized search
- test_models/ - NN models used for testing. Includes saved pre-trained CNN models and file for pre-trained weights. Also includes scripts to get the respective dataset in its appropriate form for (training and) inference.
- datasets/ - Fashion-MNIST and SVHN datasets that had to be downloaded. Scripts to get the data can be found under the respective models in test_models/

- convert_float_fixed.py - Class for quantizing an array of numbers
- evaluation_metrics.py - Functions to calculate memory consumption and cost of multiplications
- fxp_quantize.py - Using convert_float_fixed.py to fix quantization of weights and biases. Use get_fxp_model() in model_gen/ with appropriate quantization parameters to load a Keras model with quantized activations. Then pass that model to functions in fxp_quantize to quantize weights or biases. Examples shown in jupyter notebooks under analysis/
- model_data.py - functions to get properties of a model. Includes a class called Model to put together properties of a CNN model.
- requirements.txt - Python packages that were used for all the work
