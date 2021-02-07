import tensorflow as tf
import numpy as np

import keras.backend as K



class ConvertFloatFixed:
    """Class to convert floating point numbers to fixed point representations
        
    Args:
        bitwidth (integer): Bitwidth of the Fixed-point representation
        fractional_bits (integer): Fractional offset of fixed-point representation
        base (int, optional): Base for scale. Defaults to 2.
    """

    _sign_bit = 1

    def __init__(self, bitwidth, fractional_bits, base=2):
        self.base = base
        self.bitwidth = bitwidth
        self.fractional_bits = fractional_bits

    def __call__(self, input_arr):
        return self.quantize(input_arr)

    @property
    def integer_bits(self):
        """Number of integer bits in fxp-representation
        """
        return self.bitwidth - self.fractional_bits - self._sign_bit

    @property
    def scale(self):
        """Scale to shift the numbers by based on fractional offset F
        """
        return self.base ** float(self.fractional_bits)
    
    @property
    def max_value(self):
        """Maximum value based on bitwidth - 1
        """
        if self.bitwidth == self._sign_bit:
            return 0
        return (self.base **  (self.bitwidth - self._sign_bit)) - 1

    @property
    def min_value(self):
        """Minimum value based on bitwidth - 1
        """
        if self.bitwidth == self._sign_bit:
            return 0
        return - (self.base ** (self.bitwidth - self._sign_bit) - 1)

    def quantize(self, input_arr):
        """Quantize the given set of numbers to representation specified
        
        Args:
            input_arr (ndarray): input array to quantize
        Returns:
            ndarray: output array of quantized numbers for the specified representation
        """
        
        rounded_arr = np.round(input_arr * self.scale)
        clipped_arr = np.clip(rounded_arr, self.min_value, self.max_value)
        output_arr = clipped_arr / self.scale

        return output_arr
    
    def quantize_sqrt(self, input_arr):
        """Quantize the given set of numbers to representation specified 
        non-uniformly using square-root
        
        Args:
            input_arr (ndarray): input array to quantize
        Returns:
            ndarray: output array of quantized numbers for the specified representation
        """
        sqrt_arr = np.sign(input_arr) * np.sqrt(np.abs(input_arr))
        rounded_arr = np.round(sqrt_arr * self.scale)
        clipped_arr = np.clip(rounded_arr, self.min_value, self.max_value)
        unscaled_arr = clipped_arr / self.scale
        output_arr = np.sign(unscaled_arr) * (unscaled_arr ** 2)

        return output_arr


    def quantize_tf(self, x):
        """Tensorflow version of function for quantization to fixed-point
        
        Args:
            x (Tensor): Input tensor of values to quantize
        Returns:
            Tensor: Output tensor of quantized values
        """
        int_val = x * self.scale
        integer_value = K.round(int_val)
        clip_value = K.clip(integer_value, self.min_value, self.max_value)
        y = clip_value / self.scale

        return y
