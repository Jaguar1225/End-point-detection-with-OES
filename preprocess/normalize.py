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
        return (data - self.min) / (self.max - self.min)

    def standard(self, data):
        return (data - self.mean) / self.std
    


