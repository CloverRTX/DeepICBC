import numpy as np
import pandas as pd
import superp

x_var_num = superp.x_var_num

# 初始训练集生成
Xi_sample_len = superp.Xi_sample_len
Xu_sample_len = superp.Xu_sample_len
X_sample_len = superp.X_sample_len


# 针对反例附近采样
tao = superp.tao
mini_len = superp.mini_len

# 约束条件
class Sample_Handler():

    @staticmethod
    def Xi_dataSampling():

        '''初始区域的约束条件
        :return: 初始集采样
        '''
        x_data = np.linspace(-np.pi, np.pi, Xi_sample_len)
        y_data = np.linspace(-5, 5, Xi_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]


        safe = (x**2 + y**2 <= 4)


        Con = safe

        return s[Con]

    @staticmethod
    def Xi_dataSampling_Near_CounterEx(counterex):
        '''
            初始区域条件 反例附近采样
        :param counterex: 反例 [x1, x2, x3, ... , xn]
        :return: 采样点集合 ndarray
        '''

        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        safe = (x**2 + y**2 <= 4)

        Con = safe

        return np.append(s[Con], [counterex], axis=0)
        # return s[Con]

    @staticmethod
    def Xu_dataSampling():

        '''不安全区域的约束条件

        :param data: 待约束数据
        :return: 不安全集约束
        '''

        x_data = np.linspace(-np.pi, np.pi, Xu_sample_len)
        y_data = np.linspace(-5, 5, Xu_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]

        unsafe1 = x**2 + y**2<=9
        unsafe2 = x**2 + y**2 >= 2.5**2

        Con = unsafe1 & unsafe2
        return s[Con]

    @staticmethod
    def Xu_dataSampling_Near_CounterEx(counterex):
        '''
            不安全区域条件 反例附近采样
        :param counterex: 反例 [x1, x2, x3, ... , xn]
        :return: 采样点集合 ndarray
        '''

        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        unsafe1 = x ** 2 + y ** 2 <= 9
        unsafe2 = x ** 2 + y ** 2 >= 2.5 ** 2

        Con = unsafe1 & unsafe2
        # return s[Con & con]
        return np.append(s[Con], [counterex], axis=0)


    @staticmethod
    def X_dataSampling():

        '''其他区域的约束条件
            :param data: 待约束数据
            :return: 全域集约束
        '''
        x_data = np.linspace(-np.pi, np.pi, X_sample_len)
        y_data = np.linspace(-5, 5, X_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]

        con1 = x >= -np.pi
        con2 = x <= np.pi

        con3 = y >= -5
        con4 = y <= 5

        Con = con1 & con2 & con3 & con4

        return s[Con]

    @staticmethod
    def X_dataSampling_Near_CounterEx(counterex):
        '''
            全区条件 反例附近采样
        :param counterex: 反例 [x1, x2, x3, ... , xn]
        :return: 采样点集合 ndarray
        '''

        x_data = np.linspace(counterex[0] - tao, counterex[0] + tao, mini_len)
        y_data = np.linspace(counterex[1] - tao, counterex[1] + tao, mini_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, x_var_num)

        x = s[:, 0]
        y = s[:, 1]

        con1 = x >= -np.pi
        con2 = x <= np.pi

        con3 = y >= -5
        con4 = y <= 5

        Con = con1 & con2 & con3 & con4

        # return s[Con]
        return np.append(s[Con], [counterex], axis=0)


    @staticmethod
    def getTrainingData():

        # ————————————————————————————————————————————
        # 初始区域
        Xi = Sample_Handler.Xi_dataSampling()
        Xi = np.unique(Xi, axis=0)


        # ————————————————————————————————————————————
        # 不安全区域
        Xu = Sample_Handler.Xu_dataSampling()
        Xu = np.unique(Xu, axis=0)


        #——————————————————————————————————————————————————
        # 全域上的条件采点
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
    #d = np.array([[1, 2], [2, 3], [3, 4]])
    #d = np.append(d, [5, 6], axis=0)
    # print(d)
    Sample_Handler.getTrainingData()