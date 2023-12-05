import gurobipy as gp
import torch
from gurobipy import GRB
import numpy as np
import csv
import superp
from Sampling.getTrainingData import Sample_Handler
import Loss_Encoding.LossHandler as LossHandler

# debug
#m.computeIIS()
#m.write("model1.ilp")

landa = superp.LANDA
gama = superp.GAMA
x_var_num = superp.x_var_num


'''
    True : do not exist counterexamples
    False: exist counterexamples
    
    0: X0
    1: Xu
    2: X
'''

# ————————————————————————————Xu————————————————————————————
def MILP_opt_unsafeCon_block(Bx):

    m = gp.Model()


    # config
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0',lb=1,ub=3)

    m.addConstr((x0[0]-2)**2 + (x0[1]-2)**2 + (x0[2]-2)**2 + (x0[3]-2)**2 <= 1)

    y_layer = MILP_Encode_NN(m, x0, Bx)

    m.setObjective(y_layer, GRB.MINIMIZE)

    m.optimize()

    counter_ex = solution_output(m)

    if Xu_counter_ex_T_F(Bx, counter_ex):
        return True, []
    else:
        return False, counter_ex


def MILP_opt_unsafeCon(Bx):
    filename = "Sampling/SamplingData/Xu_set_data.csv"

    Tag = 0
    result0, counter_ex0 = MILP_opt_unsafeCon_block(Bx)
    if not result0:
        print(f"counterexamples : {counter_ex0}")
        Counter_Ex_Add(filename, counter_ex0, 1)
        return False, counter_ex0, Tag

    print("unsafe area passes successfully")
    return True, [], Tag + 1


# ————————————————————————————X0————————————————————————————
# x0
def MILP_opt_initCon0(Bx, Col, fx_):
    m = gp.Model()

    # config
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-0.2, ub=0.2)

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx)

    m.setObjective(y_layer_for_x0, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)



    if Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        return True, []
    else:
        return False, counter_ex

# x1
def MILP_opt_initCon1(Bx, Col, fx_):
    m = gp.Model()

    # config
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-0.2, ub=0.2)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.inf, ub=np.inf)

    m.addConstr(x1[0] ** 2 + x1[1] ** 2 + x1[2] ** 2 + x1[3] ** 2 <= 16)

    MILP_theNextPoint(m, x0, x1, Col)

    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx)

    m.setObjective(y_layer_for_x1, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)

    if Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        return True, []
    else:
        return False, counter_ex


def MILP_opt_initCon(Bx, Col, fx_):

    filename = "Sampling/SamplingData/Xi_set_data.csv"

    Tag = 0
    result0, counter_ex0 = MILP_opt_initCon0(Bx, Col, fx_)
    if not result0:
        print(f"counterexamples when k = {Tag}")
        print(f"counterexamples : {counter_ex0}")
        Counter_Ex_Add(filename, counter_ex0, 0)
        return False, counter_ex0, Tag

    Tag = 1
    result1, counter_ex1 = MILP_opt_initCon1(Bx, Col, fx_)
    if not result1:
        print(f"counterexamples when k = {Tag}")
        print(f"counterexamples : {counter_ex1}")
        Counter_Ex_Add(filename, counter_ex1, 0)
        return False, counter_ex1, Tag

    print("init area passes successfully")
    return True, [], Tag + 1


# ————————————————————————————X————————————————————————————

def MILP_opt_Indicator(Bx, Col, fx_):
    m = gp.Model()

    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0')

    m.addConstr(x0[0] ** 2 + x0[1] ** 2 + x0[2] ** 2 + x0[3] ** 2 <= 16)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1')

    m.addConstr(x1[0] ** 2 + x1[1] ** 2 + x1[2] ** 2 + x1[3] ** 2 <= 16)

    MILP_theNextPoint(m, x0, x1, Col)

    x2 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x2')

    m.addConstr(x2[0] ** 2 + x2[1] ** 2 + x2[2] ** 2 + x2[3] ** 2 <= 16)

    MILP_theNextPoint(m, x1, x2, Col)

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx, "Bx(x_0)__")
    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx, "Bx(x_1)__")
    y_layer_for_x2 = MILP_Encode_NN(m, x2, Bx, "Bx(x_2)__")

    m.addConstr(y_layer_for_x0 <= gama)
    m.addConstr(y_layer_for_x1 <= gama)

    m.setObjective(y_layer_for_x2, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)

    if X_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        return True, []
    else:
        return False, counter_ex


def MILP_opt_thirdCond(Bx, Col, fx_):
    filename = "Sampling/SamplingData/X_set_data.csv"

    Tag = 0
    result, counter_ex = MILP_opt_Indicator(Bx, Col, fx_)
    if not result:
        print(f"counterexamples : {counter_ex}")
        Counter_Ex_Add(filename, counter_ex, 2)
        return False, counter_ex, Tag

    print("state space passes successfully")
    return True, [], 1

# ——————————————————————————————support functions——————————————————————————————

def MILP_Encode_NN(m, x0, nn, str="Bx__"):
    ''' Encoding NN

    :param m: MILP Modle
    :param x0: NN input
    :param nn: NN
    :param str: name
    :return: output value
    '''
    W_b_list = nn.serveForVerify()

    length = int(len(W_b_list) / 2)

    x_layer = x0
    # x_layer.setAttr(gp.GRB.Attr.VarName, str + "X0")

    for i in range(length):
        W = W_b_list[i * 2]
        b = W_b_list[i * 2 + 1]

        # y = Wx + b
        y_layer = m.addMVar((W.shape[0],), vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf, name=str + f"Y{i + 1}")
        E = np.identity(y_layer.shape[0])
        expr = W @ x_layer + b - E @ y_layer

        name_str = f"layer_{i + 1}, y{i + 1} = W{i + 1}*x{i} + b{i + 1}"
        m.addConstr(expr == 0, name=name_str)

        if i != length - 1:
            z_layer = m.addMVar((y_layer.shape[0],), vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf,
                                name=str + f"Z{i + 1}")
            for j in range(y_layer.shape[0]):
                m.addConstr(z_layer[j] == gp.max_(y_layer[j], 0.0))
            x_layer = z_layer
        else:
            x_layer = y_layer

        m.update()

    return x_layer


def MILP_theNextPoint(m, x_pre, x_next, Col):
    alpha = superp.alpha

    u = MILP_Encode_NN(m, x_pre, Col, "Col__")[0]

    '''

    f_x = (
        "-x1 - x4 + u",
        "x1 - x2 + x1 ** 2 + u",
        "-x3 + x4 + x2 ** 2",
        "x1 - x2 - x4 + x3 ** 3 - x4 ** 3"
    )

    '''

    x0_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x1_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x2_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x3_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)

    x2_pow_2 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=np.inf)
    x3_pow_2 = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=np.inf)
    m.addConstr(x2_pow_2 == x_pre[2] * x_pre[2])
    m.addConstr(x3_pow_2 == x_pre[3] * x_pre[3])

    m.addConstr(x0_dot == -x_pre[0] - x_pre[3] + u)
    m.addConstr(x1_dot == x_pre[0] - x_pre[1] + x_pre[0] ** 2 + u)
    m.addConstr(x2_dot == -x_pre[2] + x_pre[3] + x_pre[1] ** 2)
    m.addConstr(x3_dot == x_pre[0] - x_pre[1] - x_pre[3] + x2_pow_2 * x_pre[2] - x3_pow_2 * x_pre[3])

    m.addConstr(x_next[0] == x_pre[0] + alpha * x0_dot)
    m.addConstr(x_next[1] == x_pre[1] + alpha * x1_dot)
    m.addConstr(x_next[2] == x_pre[2] + alpha * x2_dot)
    m.addConstr(x_next[3] == x_pre[3] + alpha * x3_dot)

    m.update()


def Counter_Ex_Add(filename, data, flag):
    sampleResult = []
    if flag == 0:
        sampleResult = Sample_Handler.Xi_dataSampling_Near_CounterEx(data)
    elif flag == 1:
        sampleResult = Sample_Handler.Xu_dataSampling_Near_CounterEx(data)
    else:
        sampleResult = Sample_Handler.X_dataSampling_Near_CounterEx(data)

    writeToCsv(filename, sampleResult)

def writeToCsv(filename, data):
    if len(data) == 0:
        return
    data = np.array(data).reshape((-1, x_var_num))
    num = 0
    with open(filename, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        for i in range(data.shape[0]):
            num = num + 1
            csv_write.writerow(data[i, :])
        print(f"{filename}, added {num} records")

def solution_output(m):
    print("———————————————————————————————————")
    counter_ex = []
    for v in m.getVars():
        if 'x0' in v.varName:
            counter_ex.append(v.x)
            print(f"{v.varName} = {v.x}")

        #if 'Bx' in v.varName:
        #    print(f"{v.varName} = {v.x}")

    print("optimal results: ")
    print(m.objVal)

    return counter_ex


def Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    Xi_k_point = LossHandler.calc_K_iteration(counter_ex, fx_, Col)
    loss = LossHandler.Xi_Loss_Func(Xi_k_point, Bx)
    if loss > 0:
        return False
    else:
        return True

def Xu_counter_ex_T_F(Bx, counter_ex):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    pre_y = Bx(counter_ex)
    loss = LossHandler.Xu_Loss_Func(pre_y)
    if loss > 0:
        return False
    else:
        return True

def X_counter_ex_T_F(Bx, Col, counter_ex, fx_):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    pre_z = LossHandler.Filter_Of_Loss3(counter_ex, fx_, Bx, Col)

    loss = 0.
    if pre_z.shape[0] == 0:
        loss = 0
    else:
        loss = LossHandler.X_Loss_Func(pre_z)

    if loss > 0:
        return False
    else:
        return True