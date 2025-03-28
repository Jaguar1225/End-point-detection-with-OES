import numpy as np

class SNR_Calculator:
    def __init__(self, data, term=10):
        self.data = data
        self.term = term

    def __Time_index_finder(t,T):
        try:
            return int(np.where(T.values==t)[0])
        except AttributeError:
            return int(np.where(T==t)[0])

    def Sensitivity_Factor(self, t, X, T):
        n = self.__Time_index_finder(t,T)
        temp_front_mean = X[n-self.term:n,0].mean()
        temp_back_mean = X[n:n+self.term,0].mean()
        temp_front_std = X[n-self.term:n,0].std()
        calcul = (temp_front_mean-temp_back_mean)/temp_front_std
        return abs(calcul)

    def Sensitivity_Max_Finder(self, X, T):
        Sens=np.ones((T.shape[0]))
        for n in range(T.shape[0]):
            if n<self.term:
                Sens[n]=0
            continue
        else:
            Sens[n] = self.Sensitivity_Factor(T[n],X,T)
        print(['%.1fs'%T[Sens.argmax()],'index=%d'%Sens.argmax(),Sens.max()])
        return [T[Sens.argmax()],Sens.argmax(),Sens.max()]



