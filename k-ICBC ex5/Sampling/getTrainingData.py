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
        x1_data = np.linspace(-0.2, 0.2, Xi_sample_len)
        x2_data = np.linspace(-0.2, 0.2, Xi_sample_len)
        x3_data = np.linspace(-0.2, 0.2, Xi_sample_len)
        x4_data = np.linspace(-0.2, 0.2, Xi_sample_len)

        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        return s

    @staticmethod
    def Xi_dataSampling_Near_CounterEx(counterex):
        x1_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        x2_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)
        x3_data = np.linspace(counterex[2] - tao, counterex[2] + tao, mini_len)
        x4_data = np.linspace(counterex[3] - tao, counterex[3] + tao, mini_len)

        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x1 = s[:, 0]
        x2 = s[:, 1]
        x3 = s[:, 2]
        x4 = s[:, 3]

        safe1 = x1 >= -0.2
        safe2 = x1 <= 0.2
        safe3 = x2 >= -0.2
        safe4 = x2 <= 0.2
        safe5 = x3 >= -0.2
        safe6 = x3 <= 0.2
        safe7 = x4 >= -0.2
        safe8 = x4 <= 0.2

        Con = safe1 & safe2 & safe3 & safe4 & safe5 & safe6 & safe7 & safe8

        return np.append(s[Con], [counterex], axis=0)
        # return s[Con]

    @staticmethod
    def Xu_dataSampling():
        '''
            data sample of Xu
        '''

        x1_data = np.linspace(1, 3, Xu_sample_len)
        x2_data = np.linspace(1, 3, Xu_sample_len)
        x3_data = np.linspace(1, 3, Xu_sample_len)
        x4_data = np.linspace(1, 3, Xu_sample_len)


        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x1 = s[:, 0]
        x2 = s[:, 1]
        x3 = s[:, 2]
        x4 = s[:, 3]

        unsafe = (x1 - 2) ** 2 + (x2 - 2) ** 2 + (x3 - 2) ** 2 + (x4 - 2) ** 2 <= 1.

        return s[unsafe]

    @staticmethod
    def Xu_dataSampling_Near_CounterEx(counterex):
        x1_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        x2_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)
        x3_data = np.linspace(counterex[2] - tao, counterex[2] + tao, mini_len)
        x4_data = np.linspace(counterex[3] - tao, counterex[3] + tao, mini_len)

        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x1 = s[:, 0]
        x2 = s[:, 1]
        x3 = s[:, 2]
        x4 = s[:, 3]

        unsafe = (x1 - 2) ** 2 + (x2 - 2) ** 2 + (x3 - 2) ** 2 + (x4 - 2) ** 2 <= 1.

        return np.append(s[unsafe], [counterex], axis=0)


    @staticmethod
    def X_dataSampling():
        '''
            data sample of X
        '''

        x1_data = np.linspace(-4, 4, X_sample_len)
        x2_data = np.linspace(-4, 4, X_sample_len)
        x3_data = np.linspace(-4, 4, X_sample_len)
        x4_data = np.linspace(-4, 4, X_sample_len)

        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))


        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x1 = s[:, 0]
        x2 = s[:, 1]
        x3 = s[:, 2]
        x4 = s[:, 3]

        Con = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 <= 16.

        return s[Con]

    @staticmethod
    def X_dataSampling_Near_CounterEx(counterex):

        x1_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        x2_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)
        x3_data = np.linspace(counterex[2] - tao, counterex[2] + tao, mini_len)
        x4_data = np.linspace(counterex[3] - tao, counterex[3] + tao, mini_len)

        s = np.array(np.meshgrid(x1_data, x2_data, x3_data, x4_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x1 = s[:, 0]
        x2 = s[:, 1]
        x3 = s[:, 2]
        x4 = s[:, 3]


        Con = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 <= 16.
        # return s[Con]
        return np.append(s[Con], [counterex], axis=0)


    @staticmethod
    def getTrainingData():

        # ————————————————————————————————————————————
        # X0
        Xi = Sample_Handler.Xi_dataSampling()
        Xi = np.unique(Xi, axis=0)


        # ————————————————————————————————————————————
        # Xu
        Xu = Sample_Handler.Xu_dataSampling()
        Xu = np.unique(Xu, axis=0)


        #——————————————————————————————————————————————————
        # X
        X = Sample_Handler.X_dataSampling()
        X = np.unique(X, axis=0)



        Xi_set_data = pd.DataFrame(Xi)
        Xi_set_data = Xi_set_data.round(3)

        Xu_set_data = pd.DataFrame(Xu)
        Xu_set_data = Xu_set_data.round(3)

        X_set_data = pd.DataFrame(X)
        X_set_data = X_set_data.round(3)

        Xi_set_data.to_csv("Sampling/SamplingData/Xi_set_data.csv", header=None, index=None)
        Xu_set_data.to_csv("Sampling/SamplingData/Xu_set_data.csv", header=None, index=None)
        X_set_data.to_csv("Sampling/SamplingData/X_set_data.csv", header=None, index=None)

        #print(Xi_set_data.shape[0])
        #print(Xu_set_data.shape[0])
        #print(X_set_data.shape[0])

        return Xi_set_data, Xu_set_data, X_set_data

if __name__ == "__main__":
    Sample_Handler.getTrainingData()