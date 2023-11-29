import gurobipy as gp
import torch
from gurobipy import GRB
import numpy as np
import csv
import superp
from Sampling.getTrainingData import Sample_Handler
import Loss_Encoding.LossHandler as LossHandler
import PhaseDiagram_Fx.Fx_handler as Fx_handler

# debug
#m.computeIIS()
#m.write("model1.ilp")

landa = superp.LANDA
gama = superp.GAMA
x_var_num = superp.x_var_num


'''
    True : 不存在反例
    False: 存在反例
    
    0: X0 初始区域反例
    1: Xu 不安全区域反例
    2： X 蕴含条件反例
'''

# ————————————————————————————不安全区域的验证————————————————————————————
# 不安全区域 现在不需要分块来处理了
def MILP_opt_unsafeCon_block(Bx):

    # ————————————————————————————块0————————————————————————————
    m = gp.Model()


    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.inf, ub=np.inf)

    m.addConstr(x0[0] ** 2 + x0[1] ** 2 >= 2.5 * 2.5)
    m.addConstr(x0[0] ** 2 + x0[1] ** 2 <= 9)

    y_layer = MILP_Encode_NN(m, x0, Bx)

    m.setObjective(y_layer, GRB.MINIMIZE)

    # 优化
    m.optimize()

    counter_ex = solution_output(m)


    if Xu_counter_ex_T_F(Bx, counter_ex):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex


    # 不存在反例
    #if m.objVal > landa:
    #    return True, []
    # 存在反例
    #else:
    #    return False, counter_ex

def MILP_opt_unsafeCon(Bx):
    filename = "Sampling/SamplingData/Xu_set_data.csv"

    Tag = 0
    result0, counter_ex0 = MILP_opt_unsafeCon_block(Bx)
    # 块0  有反例
    if not result0:
        print(f"有反例是在k = {Tag}")
        print(f"反例是 : {counter_ex0}")
        Counter_Ex_Add(filename, counter_ex0, 1)
        return False, counter_ex0, Tag

    # 都没有反例
    print("不安全区域条件没有反例")
    return True, [], Tag + 1


# ————————————————————————————初始区域验证————————————————————————————
# x0
def MILP_opt_initCon0(Bx, Col, fx_):
    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # Xi 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.inf, ub=np.inf)

    m.addConstr(x0[0] ** 2 + x0[1] ** 2 <= 4.)

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx)

    m.setObjective(y_layer_for_x0, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)



    if Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex


    # 不存在反例
    #if m.objVal <= gama:
    #    return True, []
    # 存在反例
    #else:
    #    return False, counter_ex

# x1
def MILP_opt_initCon1(Bx, Col, fx_):
    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # Xi 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.inf, ub=np.inf)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.inf, ub=np.inf)

    m.addConstr(x0[0] ** 2 + x0[1] ** 2 <= 4.)

    MILP_theNextPoint(m, x0, x1, Col)

    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx)

    m.setObjective(y_layer_for_x1, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)

    if Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex


    # 不存在反例
    #if m.objVal <= gama:
    #    return True, []
    # 存在反例
    #else:
    #    return False, counter_ex

# x2
def MILP_opt_initCon2(Bx, Col, fx_):
    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # Xi 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.inf, ub=np.inf)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.inf, ub=np.inf)

    x2 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x2', lb=-np.inf, ub=np.inf)

    m.addConstr(x0[0] ** 2 + x0[1] ** 2 <= 4.)

    MILP_theNextPoint(m, x0, x1, Col)

    MILP_theNextPoint(m, x1, x2, Col)

    y_layer_for_x2 = MILP_Encode_NN(m, x2, Bx)

    m.setObjective(y_layer_for_x2, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)


    if Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex

    # 不存在反例
    #if m.objVal <= gama:
    #    return True, []
    # 存在反例
    #else:
    #    return False, counter_ex

def MILP_opt_initCon(Bx, Col, fx_):

    filename = "Sampling/SamplingData/Xi_set_data.csv"
    Tag = 0
    result0, counter_ex0 = MILP_opt_initCon0(Bx, Col, fx_)

    if result0:
        Tag = 1
        result1, counter_ex1 = MILP_opt_initCon1(Bx, Col, fx_)
        if result1:
            Tag = 2
            result2, counter_ex2 = MILP_opt_initCon2(Bx, Col, fx_)
            if result2:
                print("初始区域条件没有反例")
                return True, [], 3

            # 三层 存在反例
            else:
                print(f"有反例是在k = {Tag}")
                print(f"反例是 : {counter_ex2}")
                Counter_Ex_Add(filename, counter_ex2, 0)
                return False, counter_ex2, Tag
        # 二层 存在反例
        else:
            print(f"有反例是在k = {Tag}")
            print(f"反例是 : {counter_ex1}")
            Counter_Ex_Add(filename, counter_ex1, 0)
            return False, counter_ex1, Tag
    # 一层 存在反例
    else:
        print(f"有反例是在k = {Tag}")
        print(f"反例是 : {counter_ex0}")
        Counter_Ex_Add(filename, counter_ex0, 0)
        return False, counter_ex0, Tag

# ————————————————————————————蕴含条件验证————————————————————————————

def MILP_opt_Indicator(Bx, Col, fx_):

    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # X 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0')

    x0[0].lb = -np.pi
    x0[0].ub = np.pi
    x0[1].lb = -5
    x0[1].ub = 5

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.inf, ub=np.inf)

    x1[0].lb = -np.pi
    x1[0].ub = np.pi
    x1[1].lb = -5
    x1[1].ub = 5

    MILP_theNextPoint(m, x0, x1, Col)

    x2 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x2', lb=-np.inf, ub=np.inf)
    x2[0].lb = -np.pi
    x2[0].ub = np.pi
    x2[1].lb = -5
    x2[1].ub = 5
    MILP_theNextPoint(m, x1, x2, Col)

    x3 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x3', lb=-np.inf, ub=np.inf)
    x3[0].lb = -np.pi
    x3[0].ub = np.pi
    x3[1].lb = -5
    x3[1].ub = 5
    MILP_theNextPoint(m, x2, x3, Col)

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx, "Bx(x_0)__")
    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx, "Bx(x_1)__")
    y_layer_for_x2 = MILP_Encode_NN(m, x2, Bx, "Bx(x_2)__")
    y_layer_for_x3 = MILP_Encode_NN(m, x3, Bx, "Bx(x_3)__")

    m.addConstr(y_layer_for_x0 <= gama)
    m.addConstr(y_layer_for_x1 <= gama)
    m.addConstr(y_layer_for_x2 <= gama)

    m.setObjective(y_layer_for_x3, GRB.MAXIMIZE)

    m.optimize()

    counter_ex = solution_output(m)

    if X_counter_ex_T_F(Bx, Col, counter_ex, fx_):
        # 不存在反例
        return True, []
    else:
        # 存在反例
        return False, counter_ex



    # 不存在反例
    #if m.objVal <= gama:
    #    return True, []
    # 存在反例
    #else:
    #    return False, counter_ex

def MILP_opt_thirdCond(Bx, Col, fx_):
    filename = "Sampling/SamplingData/X_set_data.csv"

    Tag = 0
    result, counter_ex = MILP_opt_Indicator(Bx, Col, fx_)
    # 存在反例
    if not result:
        print(f"有反例是在k = {Tag}")
        print(f"反例是 : {counter_ex}")
        Counter_Ex_Add(filename, counter_ex, 2)
        return False, counter_ex, Tag

    print("蕴含条件没有反例")
    return True, [], 1

# ——————————————————————————————辅助函数——————————————————————————————

# 将神经网络编码成MILP
def MILP_Encode_NN(m, x0, nn, str = "Bx__"):

    ''' MILP编码神经网络

    :param m: MILP Modle
    :param x0: NN 输入向量 （数学变量）
    :param nn: 需要编码的神经网络
    :param str: Gurobi数学变量 name
    :return: 神经网络最后的输出 （数学变量）
    '''
    W_b_list = nn.serveForVerify()

    length = int(len(W_b_list) / 2)

    x_layer = x0
    # x_layer.setAttr(gp.GRB.Attr.VarName, str + "X0")

    for i in range(length):
        W = W_b_list[i * 2]
        b = W_b_list[i * 2 + 1]

        # y = Wx + b
        y_layer = m.addMVar((W.shape[0],), vtype = GRB.CONTINUOUS, lb=-np.inf, ub=np.inf, name = str + f"Y{i + 1}")
        E = np.identity(y_layer.shape[0])
        expr = W @ x_layer + b - E @ y_layer

        name_str = f"layer_{i + 1}, y{i + 1} = W{i + 1}*x{i} + b{i + 1}"
        m.addConstr(expr == 0, name = name_str)

        if i != length - 1:
            # 添加激活函数
            z_layer = m.addMVar((y_layer.shape[0],), vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf, name = str + f"Z{i + 1}")
            for j in range(y_layer.shape[0]):
                m.addConstr(z_layer[j] == gp.max_(y_layer[j], 0.0))
            x_layer = z_layer
        else:
            x_layer = y_layer

        m.update()

    return x_layer

# 当前状态点  与  下一状态点  之间的约束建立
def MILP_theNextPoint(m, x_pre, x_next, Col):

    alpha = superp.alpha

    u = MILP_Encode_NN(m, x_pre, Col, "Col__")[0]

    '''
    
    f_x = (
        "x2",
        "-10 * sin(x1) - 0.1 * x2 + u"
    )
    
    '''

    sin_x1 = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    m.addGenConstrSin(x_pre[0], sin_x1)

    m.update()

    x0_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x1_dot = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)

    m.addConstr(x0_dot == x_pre[1])
    m.addConstr(x1_dot == -10 * sin_x1 - 0.1 * x_pre[1] + u)

    m.addConstr(x_next[0] == x_pre[0] + alpha * x0_dot)
    m.addConstr(x_next[1] == x_pre[1] + alpha * x1_dot)

    m.update()

# 反例添加
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
        print(f"{filename},更新了{num}条记录")

def solution_output(m):
    print("———————————————————————————————————")
    counter_ex = []
    for v in m.getVars():
        if 'x0' in v.varName:
            counter_ex.append(v.x)
            print(f"{v.varName} = {v.x}")

        #if 'Bx' in v.varName:
        #    print(f"{v.varName} = {v.x}")

    print("返回优化结果：")
    print(m.objVal)

    return counter_ex


def Xi_counter_ex_T_F(Bx, Col, counter_ex, fx_):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    Xi_k_point = LossHandler.calc_K_iteration(counter_ex, fx_, Col)
    loss = LossHandler.Xi_Loss_Func(Xi_k_point, Bx)
    if loss > 0:
        #确实是反例
        return False
    else:
        return True

def Xu_counter_ex_T_F(Bx, counter_ex):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    pre_y = Bx(counter_ex)
    loss = LossHandler.Xu_Loss_Func(pre_y)
    if loss > 0:
        #确实是反例
        return False
    else:
        return True

def X_counter_ex_T_F(Bx, Col, counter_ex, fx_):
    counter_ex = torch.tensor([counter_ex]).to(superp.device)
    # 第三个蕴含的条件
    pre_z = LossHandler.Filter_Of_Loss3(counter_ex, fx_, Bx, Col)

    loss = 0.
    # 如果没有满足蕴含条件的点
    #print(f"满足蕴含条件的点的个数： {pre_z.shape[0]}")
    if pre_z.shape[0] == 0:
        loss = 0
    else:
        loss = LossHandler.X_Loss_Func(pre_z)

    if loss > 0:
        #确实是反例
        return False
    else:
        return True