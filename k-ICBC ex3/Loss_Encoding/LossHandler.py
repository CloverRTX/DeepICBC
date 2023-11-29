import numpy as np
import torch
import superp
from PhaseDiagram_Fx import Fx_handler

# super.K 迭代次数
# super.step_len 步长
device = superp.device
landa = superp.LANDA
gama = superp.GAMA

def calc_K_iteration(x, fx_, control_model, k = superp.K):
    # 返回 [x0, x1, ... xk-1]

    ''' 计算fx_的k次迭代
        即，计算  x1 = x0 + f(x0, u)
                x2 = x1 + f(x1, u)
                ...
                xk = xk-1 + f(xk-1, u)

    :param x:初始点
    :param fx_: 动力系统 函数句柄
    :param control_model: 控制器 (NN)
    :param k: 迭代次数
    :return: 一个List，其中包括 k+1 个连续状态点 --> [x0,x1,...,xk-1]
    '''

    result = [x]

    for i in range(k - 1):
        # 当前点
        pre_point = result[i]

        # Xk = Xk-1 + alpha * f(Xk-1)
        # Xk = Xk-1 + alpha * dk-1

        next_point = torch.zeros_like(pre_point).to(device)
        u = control_model(pre_point)

        d_vector = Fx_handler.fx_calc_with_col(fx_, pre_point, u)

        alpha = superp.alpha

        next_point = pre_point + alpha * d_vector

        result.append(next_point)

    return result

def Xi_Loss_Func(result, Bx_model, k = superp.K):

    '''定义的Xi 的 Loss函数  (初始集)
    :param result: X_k_迭代点集
    :param model: Bx_NN
    :return: Xi的Loss
    '''

    loss1 = torch.tensor(0.)
    for i in range(k):
        data = Bx_model(result[i])
        loss1_sub = (spec_relu(data, standard = gama) - gama).sum()
        loss1 = loss1 + loss1_sub

    return loss1

def Xu_Loss_Func(pre):

    '''定义的Xu 的 Loss函数  (不安全集)
    :param pre: NN输出 (矩阵)
    :return: 返回max(-pre,0)的Loss值
    '''

    return (spec_relu(-pre, standard = -landa) + landa).sum()

# Bx第三个条件  蕴含表达式  筛选
def Filter_Of_Loss3(x_point, fx_, Bx_model, control_model, precision = 0.0):

    point_step_result = calc_K_iteration(
        x = x_point,
        fx_ = fx_,
        control_model = control_model,
        k = superp.K + 1
    )
    # [x0, x1, x2, ... xk-1] + [xk]


    num = point_step_result[0].shape[0]

    bool_index = torch.tensor([True for i in range(num)]).to(device)
    for i in range(len(point_step_result) - 1):
        point_step_result[i] = point_step_result[i].to(device)
        bool_index = bool_index & (Bx_model(point_step_result[i]) - gama - precision <= 1e-6)

    length = len(point_step_result) - 1
    target_point_step_k = point_step_result[length][bool_index]

    return Bx_model(target_point_step_k)

def X_Loss_Func(pre_x):
    return (spec_relu(pre_x, standard = gama) - gama).sum()

# 特定的Relu
def spec_relu(data = None, standard = 0.0):
    index = data > standard
    return data[index]
