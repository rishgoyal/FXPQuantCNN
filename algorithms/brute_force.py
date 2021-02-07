import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fxp_quantize
import model_data



def analyze_weights(model_obj, layer_names, bw_range, f_range, float_model_acc, 
                        fix_quant=False, parameters=None, eval_metric='acc_drop'):
    
    if fix_quant and parameters is None:
        raise ValueError('Parameters is empty')
    
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    
    for layer in layer_names:

        scores_matrix = []

        for bw in bw_range:
            scores_row = []

            for f in f_range:

                model_obj.load_model_from_path()

                if fix_quant:
                    #fix quantization for certain layers
                    model_obj = fxp_quantize.fix_weights_quantization(model_obj, parameters)

                model_obj.model = fxp_quantize.quantize_weights(model_obj.model, bw, f,
                                                            layer_name=[layer])
                scores_row.append(model_obj.evaluate_accuracy()[1])

            scores_matrix.append(scores_row)

        scores = np.array(scores_matrix)

        if layer_names.index(layer) == 0:
            model_scores = scores
        else:
            model_scores = np.dstack((model_scores, scores))
        
        print(f'Layer {layer} complete.')
    
    if eval_metric == 'acc_drop':
        model_scores = (float_model_acc - model_scores) / float_model_acc
        
    elif eval_metric == 'acc_ratio':
        model_scores = model_scores / float_model_acc
        
    elif eval_metric == 'quant_acc':
        pass
        
    return model_scores


def analyze_biases(model_obj, layer_names, bw_range, f_range, float_model_acc, fix_w_quant=False, w_parameters=None,
                                fix_b_quant=False, b_parameters=None, eval_metric='acc_drop'):
    
    if (fix_w_quant and w_parameters is None) or (fix_b_quant and b_parameters is None):
        raise ValueError('Parameters is empty')
    
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
    
    for layer in layer_names:

        scores_matrix = []

        for bw in bw_range:
            scores_row = []
            
            for f in f_range:

                model_obj.load_model_from_path()

                if fix_w_quant:
                    model_obj = fxp_quantize.fix_weights_quantization(model_obj, w_parameters)
                    
                if fix_b_quant:
                    model_obj = fxp_quantize.fix_biases_quantization(model_obj, b_parameters)

                model_obj.model = fxp_quantize.quantize_biases(model_obj.model, bw, f,
                                                            layer_name=[layer])
                scores_row.append(model_obj.evaluate_accuracy()[1])

            scores_matrix.append(scores_row)

        scores = np.array(scores_matrix)

        if layer_names.index(layer) == 0:
            model_scores = scores
        else:
            model_scores = np.dstack((model_scores, scores))
        
        print(f'Layer {layer} done.')

    if eval_metric == 'acc_drop':
        model_scores = (float_model_acc - model_scores) / float_model_acc
        
    elif eval_metric == 'acc_ratio':
        model_scores = model_scores / float_model_acc
        
    elif eval_metric == 'quant_acc':
        pass
        
    return model_scores


def analyze_activations(model_arch, model_name, test_data, layer_names, bw_range, f_range, 
                        float_model_acc, fix_quant=False, parameters=None, eval_metric='acc_drop'):
    
    if fix_quant and parameters is None:
        raise ValueError('parameters is empty')
    
    if not isinstance(layer_names, list):
        layer_names = [layer_names]
        
    for layer in layer_names:
        
        scores_matrix = []
        
        for bw in bw_range:
            scores_row = []

            for f in f_range:
                
                quant_params = {
                    layer: [bw, f]
                }
                if fix_quant:
                    quant_params = {**quant_params, **parameters}
                
                model_obj = model_data.Model(model_name, test_data, model=model_arch.get_fxp_model(quant_params))
                scores_row.append(model_obj.evaluate_accuracy()[1])
                
            scores_matrix.append(scores_row)
            
        scores = np.array(scores_matrix)

        if layer_names.index(layer) == 0:
            model_scores = scores
        else:
            model_scores = np.dstack((model_scores, scores))
        
        print(f'Layer {layer} done.')
    
    if eval_metric == 'acc_drop':
        model_scores = (float_model_acc - model_scores) / float_model_acc
        
    elif eval_metric == 'acc_ratio':
        model_scores = model_scores / float_model_acc
        
    elif eval_metric == 'quant_acc':
        pass
    
    return model_scores


def plot_results(scores, layer_names, rows, columns, figsize, bw_range, f_range, invert=False, vmax=None):
    
    if len(scores.shape) < 3:
        scores = np.expand_dims(scores, axis=3)
    fig, ax = plt.subplots(rows, columns, figsize=figsize)
    cbar_ax = fig.add_axes([.91, 0.1, .02, 0.8])
    if rows == 1 and columns == 1:
        ax = [ax]
    else:
        ax = ax.flatten()
    vmin = scores.min()
    if vmax is None:
        vmax = scores.max()
    else:
        vmax=vmax

    for i in range(scores.shape[2]):
        sns.heatmap(scores[:, :, i], annot=True, fmt='.3f', ax=ax[i], xticklabels=f_range, 
                    yticklabels=bw_range, vmin = vmin, vmax=vmax, cbar_ax=cbar_ax)
        ax[i].set_xlabel('Fractional offset')
        ax[i].set_ylabel('Bitwidth')
        ax[i].set_title(f'Layer {layer_names[i]}')
        if invert:
            ax[i].invert_yaxis()
    if rows*columns - scores.shape[2] != 0:
        for i in range(1, rows*columns - scores.shape[2] + 1):
            fig.delaxes(ax[-i])
