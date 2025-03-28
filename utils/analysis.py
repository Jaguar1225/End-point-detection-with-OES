import numpy as np
from ..preprocess import Normalize, WavelengthSelection, NoiseFilter
from ..clustering import SC, GMM, KMC

class Analysis(KMC,GMM, SC):
    def __init__(self, data_path, window_size, num_channels, model_name):
        super().__init__(data_path, window_size, num_channels)

        model_map = {
            "kmc": KMC,
            "gmm": GMM,
            "sc": SC
        }
        self.window_size = window_size
        self.num_channels = num_channels

        model_map[model_name].__init__(self.window_size, self.num_channels)

    def data_loader(self, data_path):
        self.data = np.loadtxt(data_path,delimiter=',',dtype=object)
        self.time = self.data[:,0]
        self.ees = self.data[1:,:25].astype(np.float32)
        self.oes = self.data[1:,25:].astype(np.float32)
    
    def forward(self):
        analysis_data = self.oes
        analysis_data = Normalize(analysis_data)
        analysis_data = WavelengthSelection(analysis_data)
        analysis_data = NoiseFilter(analysis_data)
        validity_score = []
        for i in range(len(self.time)-self.window_size):
            self.window_update(self.oes[i])
            if i<self.window_size:
                continue
            validity_score.append(self.validity_score())
        return validity_score



