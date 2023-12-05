import numpy as np
import sympy as sp
import torch
from PhaseDiagram_Fx import operator_sym
import superp

f_x = superp.f_x

x_var_str = superp.x_var_str
x_var_num = superp.x_var_num


control_var_str = superp.control_var_str
col_var_num = superp.col_var_num

var_num = superp.var_num

# ————————————————————————————————带控制器————————————————————————————————
def fx_incubator_with_col(fx_ = f_x, x_var = x_var_str, control_var = control_var_str, expr_type = operator_sym.operator_dict):
    '''
    :param args: fx expression
    :param x_var:
    :param control_var:
    :param expr_type: torch_operator
    :return: fx_func (fx function handle)
    '''

    var_ = x_var + ',' + control_var

    var_tuple = sp.symbols(var_)
    #print(var_tuple)

    assert(len(var_tuple) == var_num), "Variable Conflict"


    fx_expr = sp.simplify(fx_)

    print("+----------------------------------------------------------+")
    print("|Sympy expression obtained: ")
    for i in range(x_var_num):
        str(i) + "_dot"
        print(f"|\t{var_tuple[i]}_dot = {fx_expr[i]}")
    print("+----------------------------------------------------------+")

    fx_func_np = sp.lambdify(var_tuple, fx_expr, expr_type)
    #print(fx_func_np)

    return fx_func_np

def fx_calc_with_col(fx_, x_point, u_value):
    '''

    :param fx_: fx function handle
    :param x_point:
    :param u_value:
    :return: f(x, u)
    '''
    u_value = u_value.reshape((u_value.shape[0], 1))
    arr = torch.cat((x_point, u_value), 1)

    result = fx_(*(arr.t()))

    lst_result = []
    for i in range(x_var_num):
        temp = result[i].reshape((1, result[i].shape[0]))
        lst_result.append(temp)

    return torch.cat(tuple(lst_result), 0).t()