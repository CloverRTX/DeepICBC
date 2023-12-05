import numpy as np
import pandas as pd
import superp

x_var_num = superp.x_var_num


Xi_sample_len = superp.Xi_sample_len
Xu_sample_len = superp.Xu_sample_len
X_sample_len = superp.X_sample_len



tao = superp.tao
mini_len = superp.mini_len


class Sample_Handler():

    @staticmethod
    def Xi_dataSampling():
        '''
            data sample of X0
        '''

        x_data = np.linspace(-3, -1, Xi_sample_len)
        y_data = np.linspace(-3, -1, Xi_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)
        return s

    @staticmethod
    def Xi_dataSampling_Near_CounterEx(counterex):

        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        con1 = x >= -3
        con2 = x <= 1
        con3 = y >= -3
        con4 = y <= 1

        Con = con1 & con2 & con3 & con4

        return np.append(s[Con], [counterex], axis=0)
        # return s[Con]

    @staticmethod
    def Xu_dataSampling():

        '''
            data sample of Xu
        '''

        x_data = np.linspace(2, 4, Xu_sample_len)
        y_data = np.linspace(1, 3, Xu_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]

        unsafe1 = (x - 3) ** 2 + (y - 2) ** 2 <= 1

        return s[unsafe1]

    @staticmethod
    def Xu_dataSampling_Near_CounterEx(counterex):

        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        unsafe1 = (x - 3) ** 2 + (y - 2) ** 2 <= 1

        Con = unsafe1
        # return s[Con & con]
        return np.append(s[Con], [counterex], axis=0)


    @staticmethod
    def X_dataSampling():
        '''
            data sample of X
        '''
        x_data = np.linspace(-4, 4, X_sample_len)
        y_data = np.linspace(-4, 4, X_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]

        return s

    @staticmethod
    def X_dataSampling_Near_CounterEx(counterex):
        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        con1 = x >= -4
        con2 = x <= 4

        con3 = y >= -4
        con4 = y <= 4

        Con = con1 & con2 & con3 & con4

        # return s[Con]
        return np.append(s[Con], [counterex], axis=0)

    @staticmethod
    def X_bounded_area_dataSampling():
        x_data = np.linspace(-5, 5, X_sample_len)
        y_data = np.linspace(-5, 5, X_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]


        con1 = x >= -4
        con2 = x <= 4
        con3 = y >= -4
        con4 = y <= 4
        Con = con1 & con2 & con3 & con4
        return s[~Con]

    @staticmethod
    def getTrainingData():

        # ————————————————————————————————————————————
        # init area
        Xi = Sample_Handler.Xi_dataSampling()
        Xi = np.unique(Xi, axis=0)


        # ————————————————————————————————————————————
        # unsafe area
        Xu = Sample_Handler.Xu_dataSampling()
        Xu = np.unique(Xu, axis=0)


        #——————————————————————————————————————————————————
        # state space
        X = Sample_Handler.X_dataSampling()
        X = np.unique(X, axis=0)

        # ——————————————————————————————————————————————————
        # other
        X_bounded_area = Sample_Handler.X_bounded_area_dataSampling()
        X_bounded_area = np.unique(X_bounded_area, axis=0)


        Xi_set_data = pd.DataFrame(Xi)
        Xi_set_data = Xi_set_data.round(3)

        Xu_set_data = pd.DataFrame(Xu)
        Xu_set_data = Xu_set_data.round(3)

        X_set_data = pd.DataFrame(X)
        X_set_data = X_set_data.round(3)


        X_bounded_area_set_data = pd.DataFrame(X_bounded_area)
        X_bounded_area_set_data = X_bounded_area_set_data.round(3)

        Xi_set_data.to_csv("Sampling/SamplingData/Xi_set_data.csv", header=None, index=None)
        Xu_set_data.to_csv("Sampling/SamplingData/Xu_set_data.csv", header=None, index=None)
        X_set_data.to_csv("Sampling/SamplingData/X_set_data.csv", header=None, index=None)
        X_bounded_area_set_data.to_csv("Sampling/SamplingData/X_bounded_area_set_data.csv", header=None, index=None)


        return Xi_set_data, Xu_set_data, X_set_data, X_bounded_area_set_data

if __name__ == "__main__":
    Sample_Handler.getTrainingData()