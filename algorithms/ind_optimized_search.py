import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import model_data
import copy
from algorithms import fixed_bitwidth
import fxp_quantize



class IndependentOptimizedSearch:
    """(Independent Component) Optimized Search algorithm designed for finding the best bitwidth and fractional offset for all the
    layers in a given CNN. The algorithm finds optimal quantization parameters in a layer-dependent manner for each component
    (weights, biases, activations) independent of each other.

    Usage of this algorithm can be found in jupyter notebook under analysis/Layer Dependent Optimized Search and 
    analysis/Layer Independent Optimized Search
    
    Args:
        max_acc_drop_per_layer (dict): Dictionary of layer names as keys and values of acceptable loss in 
                                        inference accuracy for that layer. Order of the keys will be the order
                                        of quantization
        model_arch (Object): Model architecture - One of KerasCNN, KerasCNNLarge, InceptionCNN from model_gen
        model_name (str): Name of the model
        test_data (Tuple of ndarrays): Test data (x, y) to evaluate the network on
        float_model_acc (float): Accuracy of the original floating point network
        local_search_step_size (int, optional): Window of points to consider for local search. Defaults to 1.
        trade_off_param (float, optional): Threshold value for the algorithm to choose local search's optimum 
                                            or the believed optimum resulting from minimizing bitwidth and 
                                            fractional offset. Defaults to 0.0015.
        layer_independent_quant (bool, optional): Set to True if layers must be quantized independently of each other.
                                                    Defaults to False.
    
    """

    def __init__(self, max_acc_drop_per_layer, model_arch, model_name, test_data, float_model_acc, 
                 local_search_step_size=1, trade_off_param=0.0015, layer_independent_quant=False):
        
        self.max_acc_drop_per_layer = max_acc_drop_per_layer
        self.model_arch = model_arch
        self.model_name = model_name
        self.test_data = test_data
        self.component = None
        self.float_model_acc = float_model_acc
        self.local_search_step_size = local_search_step_size
        self.trade_off_param = trade_off_param
        self.layer_independent_quant = layer_independent_quant
        self.fix_quant_params = {}

    def evaluate_quant_performance(self, layer_name, bitwidth, frac_offset):
        """Evaluate performance of the network after quantizing the given layers
        
        Args:
            layer_name (str): Layer to quantize
            bitwidth (int): Bitwidth to quantize the layer to
            frac_offset (int): Fractional offset of the target fixed-point representation

        Raises:
            ValueError: Bitwidth is too low (less than 1)
            ValueError: Unknown component
        
        Returns:
            float: Drop in inference accuracy compared with the floating point network
        """
        self.fix_quant_params[layer_name] = [bitwidth, frac_offset]
        for k in self.fix_quant_params:
            if self.fix_quant_params[k][0] < 1:
                raise ValueError('Bitwidth is too low')
    
        if self.component == 'weights':
            model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
            model_obj = fxp_quantize.fix_weights_quantization(model_obj, self.fix_quant_params)

        elif self.component == 'biases':
            model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
            model_obj = fxp_quantize.fix_biases_quantization(model_obj, self.fix_quant_params)

        elif self.component == 'activations':
            model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_fxp_model(self.fix_quant_params))

        else:
            raise ValueError('Unknown component. Use \'weights\' , \'biases\' or \'activations\'')

        return (self.float_model_acc - model_obj.evaluate_accuracy()[1]) / self.float_model_acc

    def _quant_fixed_bw(self, start_bw):
        """Quantize the network to a fixed-bitwidth representation by calculating fractional offsets independently. 
        Used as a starting point for the optimization
        
        Args:
            start_bw (int): Starting bitwidth
        
        Returns:
            dict: Quantization parameters for all the layers in the network
        """
        
        if self.component == 'weights':
            model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
            frac_offsets = fixed_bitwidth.find_offsets_for_model_weights(model_obj, 0, start_bw)
        elif self.component == 'biases':
            model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
            frac_offsets = fixed_bitwidth.find_offsets_for_model_weights(model_obj, 1, start_bw)
        elif self.component == 'activations':
            frac_offsets = fixed_bitwidth.find_offsets_for_model_activations(self.model_arch, 
                                                                             self.model_name, 
                                                                             self.test_data, 
                                                                             start_bw)

        quant_params = {layer_name: [start_bw, f] for layer_name, f in frac_offsets.items()}
        print(quant_params)
        return quant_params
    
    def _get_initial_config(self, start_bw):
        """Get the initial configuration of quantization parameters to start optimization
        
        Args:
            start_bw (int): Starting bitwidth for the fixed-bitwidth representation
        
        Returns:
            dict: Quantization parameters for all the layers in the network
        """
        
        fxp_model_acc = self.float_model_acc - self.float_model_acc * 0.006

        while ((self.float_model_acc - fxp_model_acc)/self.float_model_acc) > 0.005:
            quant_params = self._quant_fixed_bw(start_bw)

            if self.component == 'weights':
                model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
                model_obj = fxp_quantize.fix_weights_quantization(model_obj, quant_params)
                fxp_model_acc = model_obj.evaluate_accuracy()[1]

            elif self.component == 'biases':
                model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_float_model())
                model_obj = fxp_quantize.fix_biases_quantization(model_obj, quant_params)
                fxp_model_acc = model_obj.evaluate_accuracy()[1]

            elif self.component == 'activations':
                model_obj = model_data.Model(self.model_name, self.test_data, model=self.model_arch.get_fxp_model(quant_params))
                fxp_model_acc = model_obj.evaluate_accuracy()[1]
            
            start_bw = start_bw + 4
        
        return quant_params
    
    def _reduce_bw(self, layer_name, scores_dict, max_acc_drop, bw, f, start_score):
        """Attempt to reduce the bitwidth while meeting constraints on maximum allowed drop in inference accuracy
        of the network
        
        Args:
            layer_name (str): Layer to quantize
            scores_dict (dict): Dictionary of scores
            max_acc_drop (float): Maximum allowed drop in inference accuracy for the layer
            bw (int): Bitwidth
            f (int): Fractional offset
            start_score (float): Score of the given bitwidth as a starting point

        Returns:
            int: Lowest bitwidth
            dict: Updated scores dictionary
        """
        scores = np.array([start_score])
        min_bw = bw
        
        if bw > 2:
            while (scores[0] <= max_acc_drop) and (bw > 1):
                bw = bw - 1
                s = self.evaluate_quant_performance(layer_name, bw, f)
                scores = np.insert(scores, 0, s)
        
            # find the best value
            lowest_value_index = np.where(scores <= max_acc_drop)[0][0]
            bw_arr = np.arange(bw, min_bw + 1, 1)
            min_bw = bw_arr[lowest_value_index]
        
            # add the scores to the dictionary
            for i in range(len(scores)):
                if bw_arr[i] in scores_dict:
                    scores_dict[bw_arr[i]].update({f: scores[i]})
                else:
                    scores_dict[bw_arr[i]] = {f: scores[i]}
        
        return min_bw, scores_dict
    
    def _find_min_quant_params(self, layer_name, bitwidth, frac_offset, max_acc_drop):
        """Find the lowest quantization parameters (BW, F) by simply reducing them one at a time
        
        Args:
            layer_name (str): Layer to quantize
            bitwidth (int): Starting bitwidth
            frac_offset (int): Starting fractional offset 
            max_acc_drop (float): Maximum allowed drop in inference accuracy for the layer
        
        Raises:
            ValueError: If the starting (BW, F) does not satisfy the criterion of max_acc_drop, then
                        ask for a larger starting bitwidth
        
        Returns:
            int: Minimum bitwidth
            int: Minimum fractional offset
            dict: Dictionary of all the scores
        """
        scores_dict = {}
        start_bw, start_f = bitwidth, frac_offset
        scores = np.array([self.evaluate_quant_performance(layer_name, bitwidth, frac_offset)])
        
        # Traverse diagonally until the accuracy drop is very bad
        while (scores[0] <= max_acc_drop) and (bitwidth > 1):
            bitwidth = bitwidth - 1
            frac_offset = frac_offset - 1
            s = self.evaluate_quant_performance(layer_name, bitwidth, frac_offset)
            scores = np.insert(scores, 0, s)
        
        # one additional evaluation to avoid local minimum unless no bits remain
        if bitwidth > 1:
            bitwidth = bitwidth - 1
            frac_offset = frac_offset - 1
            s = self.evaluate_quant_performance(layer_name, bitwidth, frac_offset)
            scores = np.insert(scores, 0, s)
        
        try:
            lowest_value_index = np.where(scores <= max_acc_drop)[0][0]
        except:
            print('No suitable points found. Try a larger starting bitwidth or increase' + \
                    f'the max_acc_drop threshold for layer {layer_name}')
            print(f'max_acc_drop: {max_acc_drop:.6f} | collected values: {scores}' + \
                 f'| bitwidths: {np.arange(bitwidth, start_bw + 1, 1)}, fractional_offsets: {np.arange(frac_offset, start_f + 1, 1)}')
            return 0, 0, 0

        bw_arr = np.arange(bitwidth, start_bw + 1, 1)
        f_arr = np.arange(frac_offset, start_f + 1, 1)
        min_bw, min_f = bw_arr[lowest_value_index], f_arr[lowest_value_index]
        
        # add scores to dictionary of scores
        for i in range(len(scores)):
            scores_dict[bw_arr[i]] = {f_arr[i]: scores[i]}
        
        # Attempt to reduce the bitwidth further
        # if possible
        min_bw, scores_dict = self._reduce_bw(layer_name, scores_dict, max_acc_drop, min_bw, min_f, 
                                              scores[lowest_value_index])

        return min_bw, min_f, scores_dict
    
    def _search_local(self, scores_dict, layer_name, bw, f):
        """Search for an optimum in a local area after finding minimum bitwidth and fractional offset
        
        Args:
            scores_dict (dict): Dictionary to store scores for calculated accuracy drops for (BW, F) 
            as returned from _find_min_quant_params
            layer_name (str): Layer to quantize
            bw (int): Bitwidth
            f (int): Fractional offset
        
        Returns:
            dict: Dictionary of scores
            int: Optimal bitwidth
            int: Optimal Fractional offset
        """
        candidate_points = np.ones((self.local_search_step_size + 1 + 1, self.local_search_step_size + 1 + 1))
        
        for i, j in itertools.product(np.arange(-1, self.local_search_step_size + 1, 1).tolist(), 
                                      repeat=2):
            # if bw = 1, then we cannot evaluate anything lower
            if (bw + i < 1):
                continue
            
            if (bw + i) in scores_dict:
                # if bw exists
                if ((f + j) not in scores_dict[bw + i]):
                    # if f does not exist, add it to the dict of the corresponding bw
                    scores_dict[bw + i][f + j] = \
                            self.evaluate_quant_performance(layer_name, bw + i, f + j)
            
            else:
                # if bw does not exist, add it and the corresponding f to the dict
                scores_dict[bw + i] = {
                    f + j: self.evaluate_quant_performance(layer_name, bw + i, f + j)
                }

            candidate_points[i + 1][j + 1] = scores_dict[bw + i][f + j]
        
        # search for local optimum
        a, b = np.where(candidate_points == candidate_points.min())
        opt_bw, opt_f = bw + a[0] - 1, f + b[0] - 1
        
        return scores_dict, opt_bw, opt_f
    
    def run(self, component, start_bw=8):
        """Run the algorithm
        
        Args:
            component (str): Weights, activations or Biases to quantize
            start_bw (int, optional): Starting bitwidth. Defaults to 8.
        
        Returns:
            dict: Dictionary of scores for each layer
            dict: Dictionary of optimal quantization parameters for each layer
            dict: Accuracy drops after quantizing each successive layer
        """
        
        scores = {}
        opt_params = {}
        seq_acc_drop = {}
        self.component = component
        start_params = self._get_initial_config(start_bw)

        if self.layer_independent_quant:
            print('Finding quantization for each layer independently of other layers')
        else:
            print('Finding quantization for each layer depending on quantization for previous layers')
        
        for layer_name in self.max_acc_drop_per_layer:

            start_bw, start_f = start_params[layer_name][0], start_params[layer_name][1]
            
            print(f'Quantizing layer {layer_name}')
            max_acc_drop = self.max_acc_drop_per_layer[layer_name]

            if not self.layer_independent_quant:
                self.fix_quant_params = copy.deepcopy(opt_params)
            else:
                self.fix_quant_params = {}
                
            # Find min BW and F by traversing the plane diagonally (global search)
            min_bw, min_f, scores_dict = self._find_min_quant_params(layer_name, start_bw, start_f, max_acc_drop)
            if (min_bw == 0) and (min_f == 0) and (scores_dict == 0):
                return 0, 0, 0
            # run a local search and return the point with the minimum accuracy drop
            scores_dict, opt_bw, opt_f = self._search_local(scores_dict, layer_name, min_bw, min_f)
            scores_diff = scores_dict[min_bw][min_f] - scores_dict[opt_bw][opt_f]
            
            print(f'Global opt: {(min_bw, min_f)}', f'Local opt: {(opt_bw, opt_f)}', f'Performance diff: {scores_diff:.6f}')
            # If local search returns a different result than global search
            if not ((opt_bw == min_bw) and (opt_f == min_f)):
                # if this difference is larger than a threshold (tunable trade off parameter)
                if np.abs(scores_diff) >= self.trade_off_param:
                    # trust global search if it's better
                    if scores_dict[min_bw][min_f] < scores_dict[opt_bw][opt_f]:
                        opt_bw, opt_f = min_bw, min_f
                else:
                    # if the difference isnt large enough
                    # choose the option with the lowest bitwidth
                    # if there is also a difference in bitwidth
                    if min_bw < opt_bw:
                        opt_bw, opt_f = min_bw, min_f
            
            print('Chosen: ', (opt_bw, opt_f))
            
            scores[layer_name] = scores_dict
            opt_params[layer_name] = (opt_bw, opt_f)
            if not self.layer_independent_quant:
                accuracy_drop = scores_dict[opt_bw][opt_f]
            else:
                self.fix_quant_params = copy.deepcopy(opt_params)
                accuracy_drop = self.evaluate_quant_performance(layer_name, opt_bw, opt_f)
            seq_acc_drop[layer_name] = accuracy_drop
            
            print(f'After quantizing layer {layer_name} | Measured accuracy drop {accuracy_drop:.6f} ' + \
                  f'| Acceptable accuracy drop: {max_acc_drop:.6f}')

        return scores, opt_params, seq_acc_drop
    
    @staticmethod
    def plot_scores_matrix(scores_dict, rows, columns, figsize):
        """Plot the scores collected for each layer
        
        Args:
            scores_dict (dict): Dictionary with the scores for each layer {layer_name: {bw: {f: __}}}
            rows (int): Number of rows for the plot
            columns (int): Number of columns for the plot
            figsize (tuple): Tuple of the size of the figure
        """

        layer_dfs = []
        for layer_name in scores_dict:
            s = scores_dict[layer_name]
            df = pd.DataFrame.from_dict({i: s[i] for i in s.keys()}, orient='index')
            df = df.sort_index(axis = 0)
            df = df.sort_index(axis = 1)
            layer_dfs.append(df)
        
        fig, axes = plt.subplots(rows, columns, figsize=figsize)
        ax = axes.flat
        if rows*columns - len(list(scores_dict.keys())) != 0:
            for i in range(1, rows*columns - len(list(scores_dict.keys())) + 1):
                fig.delaxes(ax[-i])

        for df, i in zip(layer_dfs, np.arange(len(layer_dfs))):
            sns.heatmap(df, annot=True, fmt='.3f', ax=ax[i])
            ax[i].set_xlabel('Fractional offset')
            ax[i].set_ylabel('Bitwidth')
            ax[i].set_title(f'Layer {list(scores_dict.keys())[i]}')