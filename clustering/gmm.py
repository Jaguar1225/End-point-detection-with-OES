import numpy as np
from window import Window

class GMM(Window):
    def __init__(self, window_size, iter_max=100):
        super().__init__(window_size)
        self.iter_max = iter_max

    def forward(self):

        cluster_1 = self.window[:len(self.window)//2]
        cluster_2 = self.window[len(self.window)//2:]

        mu_1, cov_1 = cluster_1.mean(axis=0), self.covariance(cluster_1)
        mu_2, cov_2 = cluster_2.mean(axis=0), self.covariance(cluster_2)

        prev_label = np.zeros(len(self.window))

        for i in range(self.iter_max):
            current_label = np.argmax(
                np.c_[
                    self.gaussian_pdf(cluster_1, mu_1, cov_1),
                    self.gaussian_pdf(cluster_2, mu_2, cov_2)
                ],
                axis=1
            )
            
            if (i > 0)&(current_label == prev_label).all():
                break

            prev_label = current_label.copy()

            mu_1 = (cluster_1*current_label).sum(axis=0)/current_label.sum()
            mu_2 = (cluster_2*current_label).sum(axis=0)/current_label.sum()

            cov_1 = self.covariance(cluster_1-mu_1)
            cov_2 = self.covariance(cluster_2-mu_2)

        return self.validity_score()
            
    
    def gaussian_pdf(self, x, mean, cov):
        dim = x.shape[1]
        return (1/(2*np.pi*cov**2)**(1/dim))*np.exp(-(x-mean)**2/(2*cov**2))
    
    def covariance(self, X: np.ndarray):
        '''
        X: np.ndarray (n, d)
        '''
        return np.sqrt(((X-X.mean(axis=0, keepdims=True))**2).sum(axis=0)/(len(X)-1))
        
        
