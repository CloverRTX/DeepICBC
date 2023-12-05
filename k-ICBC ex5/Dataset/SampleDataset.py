import copy
import csv
import random

import numpy as np
import math
import torch

class Dataset():

    def __init__(self, path_Xi, path_Xu, path_X):

        with open(path_Xi, "r") as fp:
            data = list(csv.reader(fp))
            self.data_Xi = np.array(data).astype(float)

        with open(path_Xu, "r") as fp:
            data = list(csv.reader(fp))
            self.data_Xu = np.array(data).astype(float)

        with open(path_X, "r") as fp:
            data = list(csv.reader(fp))
            self.data_X = np.array(data).astype(float)

        self.getLen()

    def reload(self, path_Xi, path_Xu, path_X):

        with open(path_Xi, "r") as fp:
            data = list(csv.reader(fp))
            self.data_Xi = np.array(data).astype(float)

        with open(path_Xu, "r") as fp:
            data = list(csv.reader(fp))
            self.data_Xu = np.array(data).astype(float)

        with open(path_X, "r") as fp:
            data = list(csv.reader(fp))
            self.data_X = np.array(data).astype(float)


    def batchDivision(self, Xi_batch_size, Xu_batch_size, X_batch_size):

        temp_data_Xi = self.arr_split(self.data_Xi, Xi_batch_size)
        temp_data_Xu = self.arr_split(self.data_Xu, Xu_batch_size)
        temp_data_X = self.arr_split(self.data_X, X_batch_size)

        num_max = max(len(temp_data_Xi), len(temp_data_Xu))
        num_max = max(num_max, len(temp_data_X))

        # X0
        num_init = len(temp_data_Xi)
        for i in range(num_max - num_init):
            temp = copy.deepcopy(temp_data_Xi[i])
            temp_data_Xi.append(temp)

        # Xu
        num_unsafe = len(temp_data_Xu)
        for i in range(num_max - num_unsafe):
            temp = copy.deepcopy(temp_data_Xu[i])
            temp_data_Xu.append(temp)

        # X
        num_X = len(temp_data_X)
        for i in range(num_max - num_X):
            temp = copy.deepcopy(temp_data_X[i])
            temp_data_X.append(temp)

        return temp_data_Xi, temp_data_Xu, temp_data_X

    def fullBatch(self):
        return [copy.deepcopy(self.data_Xi)], [copy.deepcopy(self.data_Xu)], \
               [copy.deepcopy(self.data_X)]

    def arr_split(self, data, line_num):
        total_line = data.shape[0]
        yu = total_line % line_num
        new_data = copy.deepcopy(data[0 : total_line - yu])
        last_data = copy.deepcopy(data[total_line - yu :, :])
        if total_line // line_num > 0:
            lst = np.array_split(new_data, total_line // line_num, axis = 0)
            if yu > 0:
                lst.append(last_data)
            return lst
        else:
            return [last_data]

    def getLen(self):
        print(f"The line number of Xi : {self.data_Xi.shape[0]}")
        print(f"The line number of Xu : {self.data_Xu.shape[0]}")
        print(f"The line number of X : {self.data_X.shape[0]}")
