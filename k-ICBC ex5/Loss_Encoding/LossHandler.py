import torch
import superp
from PhaseDiagram_Fx import Fx_handler

device = superp.device
landa = superp.LANDA
gama = superp.GAMA

def calc_K_iteration(x, fx_, control_model, k = superp.K):
    # return [x0, x1, ... xk-1]

    ''' f^0, f^1, f^2, ... ,f^(k-1)
        calc  x1 = x0 + f(x0, u) * alpha
              x2 = x1 + f(x1, u) * alpha
              ...
              xk = xk-1 + f(xk-1, u) * alpha

    :param x: x0
    :param fx_: fx function handle
    :param control_model:
    :param k:
    :return: list
    '''

    result = [x]

    for i in range(k - 1):
        pre_point = result[i]

        next_point = torch.zeros_like(pre_point).to(device)
        u = control_model(pre_point)

        d_vector = Fx_handler.fx_calc_with_col(fx_, pre_point, u)

        alpha = superp.alpha

        next_point = pre_point + alpha * d_vector

        result.append(next_point)

    return result

def Xi_Loss_Func(result, Bx_model, k = superp.K):
    loss1 = torch.tensor(0.)
    for i in range(k):
        data = Bx_model(result[i])
        loss1_sub = (spec_relu(data, standard = gama) - gama).sum()
        loss1 = loss1 + loss1_sub

    return loss1

def Xu_Loss_Func(pre):
    return (spec_relu(-pre, standard = -landa) + landa).sum()

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
        bool_index = bool_index & (Bx_model(point_step_result[i]) - gama - precision <= 1e-8)

    length = len(point_step_result) - 1
    target_point_step_k = point_step_result[length][bool_index]

    return Bx_model(target_point_step_k)

def X_Loss_Func(pre_x):
    return (spec_relu(pre_x, standard = gama) - gama).sum()

def spec_relu(data = None, standard = 0.0):
    index = data > standard
    return data[index]
