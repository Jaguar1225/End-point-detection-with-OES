import numpy as np

class Normalize:
    def __init__(self, **params):
        self.nor_fns = {
            "minmax": self.minmax,
            "standard": self.standard
        }
        self.normalize_fn = self.nor_fns[params["normalize_method"]]

    def normalize(self, data):
        return self.normalize_fn(data)
    
    def minmax(self, data):
        return (data - data.min(axis=0, keepdims=True)) / (data.max(axis=0, keepdims=True) - data.min(axis=0, keepdims=True))

    def standard(self, data):
        return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, keepdims=True)
    


