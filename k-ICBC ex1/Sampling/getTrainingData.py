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
        x_data = np.linspace(-3, -1, Xi_sample_len)
        y_data = np.linspace(-3, -1, Xi_sample_len)

        s = np.array(np.meshgrid(x_data, y_data))

        b = s.reshape(-1, order='F')

        s = b.reshape(-1, 2)

        x = s[:, 0]
        y = s[:, 1]

        return s

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

        con1 = x >= -3
        con2 = x <= 1
        con3 = y >= -3
        con4 = y <= 1

        Con = con1 & con2 & con3 & con4

        return np.append(s[Con], [counterex], axis=0)
        # return s[Con]

    @staticmethod
    def Xu_dataSampling():

        '''不安全区域的约束条件

        :param data: 待约束数据
        :return: 不安全集约束
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

        unsafe1 = (x - 3) ** 2 + (y - 2) ** 2 <= 1

        Con = unsafe1
        # return s[Con & con]
        return np.append(s[Con], [counterex], axis=0)


    @staticmethod
    def X_dataSampling():

        '''其他区域的约束条件
            :param data: 待约束数据
            :return: 全域集约束
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

        con1 = x >= -4
        con2 = x <= 4

        con3 = y >= -4
        con4 = y <= 4

        Con = con1 & con2 & con3 & con4

        # return s[Con]
        return np.append(s[Con], [counterex], axis=0)

    @staticmethod
    def X_bounded_area_dataSampling():

        '''其他区域的约束条件
            :param data: 待约束数据
            :return: 全域集约束
        '''
        x_data = np.linspace(-4.2, 4.2, X_sample_len)
        y_data = np.linspace(-4.2, 4.2, X_sample_len)

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

        # ——————————————————————————————————————————————————
        # 边界域上的条件采点
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
    #d = np.array([[1, 2], [2, 3], [3, 4]])
    #d = np.append(d, [5, 6], axis=0)
    # print(d)
    Sample_Handler.getTrainingData()