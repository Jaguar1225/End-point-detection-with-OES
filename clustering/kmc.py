import numpy as np
from window import Window

class KMC(Window):
    def __init__(self, window_size, iter_max=100):
        super().__init__(window_size)
        self.iter_max = iter_max
    
    def forward(self):
        center_cluster_1 = self.window[0:1]
        center_cluster_2 = self.window[-1:]

        prev_label = np.zeros(len(self.window))

        for i in range(self.iter_max):
            dist_1 = np.linalg.norm(self.window - center_cluster_1, axis=1)
            dist_2 = np.linalg.norm(self.window - center_cluster_2, axis=1)

            current_label = np.argmin(np.c_[dist_1, dist_2], axis=1)

            if (i > 0)&(current_label == prev_label).all():
                break

            prev_label = current_label.copy()

            center_cluster_1 = self.window[current_label==0].mean(axis=0,keepdims=True)
            center_cluster_2 = self.window[current_label==1].mean(axis=0,keepdims=True)

        return self.validity_score()

            
        
        
