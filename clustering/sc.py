import numpy as np
from .window import Window

'''
S. Lee, H. Choi, J. Kim, H. Chae, Plasma. Process. Polym. 20(6) (2023)

doi: 10.1002/ppap.202200238
'''

class SC(Window):
    def __init__(self, window_size, num_channels):
        super().__init__(window_size, num_channels)

    def __call__(self):
        return self.forward()

    def forward(self):
        distance_matrix = self.distance_matrix(self.window)

        A = np.exp(-distance_matrix)
        D = np.eye(len(distance_matrix))*A.sum(axis=1)
        L = D-A

        NorL = D**(-1/2)@L@D**(-1/2)

        eig_val, eig_vec = np.linalg.eig(NorL)

        label = eig_vec[eig_vec.argsort()[1]]

        label[label>0] = 1
        label[label<0] = 0

        self.label = label

        return self.validity_score()
        
    def distance_matrix(self, X):
        return (X**2).sum(axis=1) - 2*X@X.T + (X.T**2).sum(axis=0)


