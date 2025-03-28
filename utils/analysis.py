import numpy as np
from ..preprocess import Normalize, WavelengthSelection, NoiseFilter
from ..clustering import SC, GMM, KMC
from .plot.plotter import Plotter
from glob import glob

class Analysis(KMC,GMM, SC, Normalize, WavelengthSelection, NoiseFilter):
    def __init__(self, **params):

        model_map = {
            "kmc": KMC,
            "gmm": GMM,
            "sc": SC
        }
        self.window_size = params["window_size"]
        self.num_channels = params["num_channels"]

        self.data_loader(params["data_path"])

        model_map[params["model_name"]].__init__(self.window_size, self.num_channels)

        Normalize.__init__(self, **params)
        WavelengthSelection.__init__(self, **params)
        NoiseFilter.__init__(self, **params)

    def data_loader(self, data_path: str) -> None:
        self.data = {}
        for file_path in glob(f"{data_path}/*.csv"):
            file_name = file_path.split("/")[-1].split(".")[0]
            temp_load = np.loadtxt(file_path,delimiter=',',dtype=object)
            self.data[file_name] = {}
            self.data[file_name]["time"] = temp_load[:,0]
            self.data[file_name]["ees"] = temp_load[1:,:25].astype(np.float32)
            self.data[file_name]["oes"] = temp_load[1:,25:].astype(np.float32)
    
    def analysis(self) -> None:
        for file_name, file_data in self.data.items():
            oes_selected = self.wavelength_selection(file_data["oes"])
            oes_normalized = self.normalize(oes_selected)
            oes_denoised = self.denoise(oes_normalized)

            validity_score = []

            for i in range(len(file_data["time"])-self.window_size):
                self.window_update(oes_denoised[i])
                if i<self.window_size:
                    continue
                validity_score.append(self.validity_score())
        
            plotter = Plotter()
            plotter.plot_line(
                file_data["time"][self.window_size:], 
                validity_score, 
                title="Validity Score", 
                xlabel="Time", 
                ylabel="Validity Score",
                save_name=f"{file_name}.png"
                )



