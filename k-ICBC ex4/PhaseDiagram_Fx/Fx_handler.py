import numpy as np
import sympy as sp
import torch
from PhaseDiagram_Fx import operator_sym
import superp

# 向量场
f_x = superp.f_x

# 状态——变量串
x_var_str = superp.x_var_str
# 状态——变量个数
x_var_num = superp.x_var_num

# 控制——变量串
control_var_str = superp.control_var_str
# 控制——变量个数
col_var_num = superp.col_var_num

# 变量总个数
var_num = superp.var_num

# ————————————————————————————————带控制器————————————————————————————————
def fx_incubator_with_col(fx_ = f_x, x_var = x_var_str, control_var = control_var_str, expr_type = operator_sym.operator_dict):
    '''
        产生 fx 表达式

    :param args: fx表达式
    :param x_var: 状态_变量串
    :param control_var: 控制_变量串
    :param expr_type: 在此基础上转化sympy表达式，默认是numpy
    :return: fx_func

    '''

    var_ = x_var + ',' + control_var

    # 变量元组
    var_tuple = sp.symbols(var_)
    #print(var_tuple)

    assert(len(var_tuple) == var_num), "变量个数不一致"


    # string 转 sympy表达式
    fx_expr = sp.simplify(fx_)

    # 输出，检验转换是否正确
    print("+----------------------------------------------------------+")
    print("|转换后的Sympy表达式 ：")
    for i in range(x_var_num):
        str(i) + "_dot"
        print(f"|\t{var_tuple[i]}_dot = {fx_expr[i]}")
    print("+----------------------------------------------------------+")

    # sympy表达式转numpy函数，加快计算
    fx_func_np = sp.lambdify(var_tuple, fx_expr, expr_type)
    #print(fx_func_np)

    return fx_func_np

def fx_calc_with_col(fx_, x_point, u_value):
    '''

    :param fx_: 向量场
    :param x_point: 传入的状态值，X点的坐标
                一般是一个tensor数组
                一行代表一个状态X的坐标
                列数表示X的维度
    :return: 计算结果
            一般是一个tensor数组
            一行代表状态X的向量场
            每列代表对应分量

    '''
    # arr = torch.tensor([[1., 2, 3], [2, 3, 4], [3, 4, 5]], requires_grad = True)
    u_value = u_value.reshape((u_value.shape[0], 1))
    arr = torch.cat((x_point, u_value), 1)

    result = fx_(*(arr.t()))

    lst_result = []
    for i in range(x_var_num):
        temp = result[i].reshape((1, result[i].shape[0]))
        lst_result.append(temp)

    return torch.cat(tuple(lst_result), 0).t()

# 待考虑传入的是[1, 2, 3]一维的情况

# ————————————————————————————————不带控制器————————————————————————————————

def fx_incubator(*args, x_var = "x1,x2", expr_type = operator_sym.operator_dict):
    '''
        产生 fx 表达式

    :param args: fx表达式
    :param x_var: 状态_变量串
    :param expr_type: 在此基础上转化sympy表达式，默认是numpy
    :return: fx_func

    '''

    # 变量元组
    var_tuple = sp.symbols(x_var)
    assert(len(var_tuple) == var_num), "变量个数不一致"

    # string 转 sympy表达式
    fx_expr = sp.simplify(*args)

    # 输出，检验转换是否正确
    print("+----------------------------------------------------------+")
    print("|转换后的Sympy表达式 ：")
    for i in range(var_num):
        str(i) + "_dot"
        print(f"|\t{var_tuple[i]}_dot = {fx_expr[i]}")
    print("+----------------------------------------------------------+")

    # sympy表达式转numpy函数，加快计算
    fx_func_np = sp.lambdify(var_tuple, fx_expr, expr_type)

    return fx_func_np

def fx_calc(fx_, x_point):
    '''

    :param fx_: 向量场
    :param x_point: 传入的状态值，X点的坐标
                一般是一个tensor数组
                一行代表一个状态X的坐标
                列数表示X的维度
    :return: 计算结果
            一般是一个tensor数组
            一行代表状态X的向量场
            每列代表对应分量

    '''
    assert(x_point.shape[1] == var_num), "状态点维度与向量场表达式不一致"
    result = fx_(*np.transpose(x_point))
    return np.column_stack(result)


