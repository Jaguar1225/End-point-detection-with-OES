import numpy as np
import pywt

class NoiseFilter:
    def __init__(self, **params):
        self.filter_fns = {
            "lowpass": self.lowpass,
            "moving_average": self.moving_average
        }
        self.filter_fn = self.filter_fns[params["filter_method"]]

    def filter(self, data):
        return self.filter_fn(data)

    def lowpass(self, data, threshold=0.0003, wavelet="coif17", DWT_level=1):
        coeff = pywt.wavedec2(data, wavelet, mode="sym", level=DWT_level)

        for i in range(1, len(coeff)):
            coeff_layer_list = list(coeff[i])
            coeff_layer_list_shape = coeff_layer_list[1].shape
            coeff_layer_list[1] = np.ones(coeff_layer_list_shape)
            coeff[i] = tuple(coeff_layer_list)

        data = pywt.waverec2(coeff, wavelet, mode="sym")[:,:data.shape[1]]
        return data
    
    def moving_average(self, data, window=3):
        return np.convolve(data, np.ones(window)/window, mode="valid")
    
    
    
