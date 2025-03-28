import numpy as np

class Window:
    def __init__(self, window_size, num_channels):
        self.window_size = window_size
        self.num_channels = num_channels
        self.window = np.empty(shape = (window_size, num_channels))

    def validity_score(self):
        center_cluter_1 = np.mean(self.window[self.label==0], axis = 0)
        center_cluter_2 = np.mean(self.window[self.label==1], axis = 0)

        dist_1 = np.linalg.norm(self.window[self.label==0] - center_cluter_1,axis=1)
        dist_2 = np.linalg.norm(self.window[self.label==1] - center_cluter_2, axis=1)
        dist_overall = np.linalg.norm(self.window-self.window.mean(axis=0),axis=1)

        return (dist_1.sum() + dist_2.sum()) / dist_overall.sum()
    
    def window_update(self, data):
        self.window = np.append(self.window, data, axis=0)
        self.window = self.window[1:]
