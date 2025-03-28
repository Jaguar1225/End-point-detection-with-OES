import numpy as np
from preprocess import Normalize, WavelengthSelection, NoiseFilter
from clustering import SC, GMM, KMC
from .plot.plotter import Plotter
from glob import glob
from tqdm import tqdm

class Analysis(KMC,GMM, SC, Normalize, WavelengthSelection, NoiseFilter):
    def __init__(self, **params):
        self.params = params
        self.model_map = {
            "kmc": KMC,
            "gmm": GMM,
            "sc": SC
        }
        self.window_size = params["window_size"]
        self.num_channels = params["num_channels"]

        self.data_loader(params["data_path"])

        

        Normalize.__init__(self, **params)
        WavelengthSelection.__init__(self, **params)
        NoiseFilter.__init__(self, **params)

    def data_loader(self, data_path: str) -> None:
        self.data = {}
        pbar = tqdm(glob(f"{data_path}/*.csv"), desc="Loading data",leave=False)
        for file_path in pbar:
            file_name = file_path.split("/")[-1].split(".")[0]
            temp_load = np.loadtxt(file_path,delimiter=',',dtype=str)
            start_idx = np.where(temp_load[:,0] == "DATE")[0][0]
            self.data[file_name] = {}
            self.data[file_name]["time"] = temp_load[start_idx+1:,0]
            self.data[file_name]["eeprom"] = temp_load[start_idx,26:].astype(np.float32)
            self.data[file_name]["ees"] = temp_load[start_idx+1:,1:26].astype(np.float32)
            self.data[file_name]["oes"] = temp_load[start_idx+1:,26:].astype(np.float32)
    
    def analysis(self) -> None:
        pbar = tqdm(self.data.items(), desc="Processing",leave=False)
        for file_name, file_data in pbar:
            oes_selected = self.selection(file_data["eeprom"],file_data["oes"])
            oes_normalized = self.normalize(oes_selected)
            oes_denoised = self.filter(oes_normalized)

            self.model_map[self.params["model_name"].lower()].__init__(self, self.window_size, oes_denoised.shape[-1])

            validity_score = []

            pbar_time = tqdm(range(len(file_data["time"])-self.window_size), desc="Processing",leave=False)
            for i in pbar_time:
                self.window_update(oes_denoised[i:i+1])
                if i<self.window_size:
                    continue
                validity_score.append(self())
        
            plotter = Plotter()
            plotter.plot_line(
                file_data["time"][self.window_size:], 
                validity_score, 
                title="Validity Score", 
                xlabel="Time", 
                ylabel="Validity Score",
                save_name=f"{file_name}.png"
                )



