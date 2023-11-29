import gurobipy as gp
from gurobipy import GRB
import numpy as np
import csv

# debug

#m.computeIIS()
#m.write("model1.ilp")

#print(m.FarkasDual)
#print(m.FarkasProof)


# 不安全区域的验证
def MILP_opt_unsafeCon(Bx):

    eye1 = False
    eye2 = False
    eye3 = False
    eye4 = False


    # ————————————————————————————块1————————————————————————————
    m1 = gp.Model()
    m1.setParam('Outputflag', 0)

    # 配置
    # m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    x0 = m1.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi / 2, ub=np.pi / 2)


    m1.addConstr(x0[0] >= -np.pi / 2)
    m1.addConstr(x0[0] <= -np.pi / 6)

    m1.addConstr(x0[1] >= -np.pi / 2)
    m1.addConstr(x0[1] <= np.pi / 2)

    y_layer = MILP_Encode_NN(m1, x0, Bx)

    m1.setObjective(y_layer, GRB.MINIMIZE)

    # 优化
    m1.optimize()

    x0_result = solution_output(m1, 0)
    if len(x0_result) != 0:
        eye1 = False
    else:
        eye1 = True
    filename = "Sampling/SamplingData/Xu_set_data.csv"
    writeToCsv(filename, x0_result)



    # ————————————————————————————块2————————————————————————————
    m2 = gp.Model()
    m2.setParam('Outputflag', 0)

    x0_len = Bx.input_dim

    x0 = m2.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi / 2, ub=np.pi / 2)

    m2.addConstr(x0[0] >= -np.pi / 6)
    m2.addConstr(x0[0] <= np.pi / 6)

    m2.addConstr(x0[1] >= -np.pi / 2)
    m2.addConstr(x0[1] <= -np.pi / 6)

    y_layer = MILP_Encode_NN(m2, x0, Bx)

    m2.setObjective(y_layer, GRB.MINIMIZE)

    # 优化
    m2.optimize()

    x0_result = solution_output(m2, 0)
    if len(x0_result) != 0:
        eye2 = False
    else:
        eye2 = True
    writeToCsv(filename, x0_result)

    # ————————————————————————————块3————————————————————————————
    m3 = gp.Model()
    m3.setParam('Outputflag', 0)

    x0_len = Bx.input_dim

    x0 = m3.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi / 2, ub=np.pi / 2)

    m3.addConstr(x0[0] >= -np.pi / 6)
    m3.addConstr(x0[0] <= np.pi / 6)

    m3.addConstr(x0[1] >= np.pi / 6)
    m3.addConstr(x0[1] <= np.pi / 2)

    y_layer = MILP_Encode_NN(m3, x0, Bx)

    m3.setObjective(y_layer, GRB.MINIMIZE)

    # 优化
    m3.optimize()

    x0_result = solution_output(m3, 0)
    if len(x0_result) != 0:
        eye3 = False
    else:
        eye3 = True
    writeToCsv(filename, x0_result)

    # ————————————————————————————块4————————————————————————————
    m4 = gp.Model()
    m4.setParam('Outputflag', 0)

    x0_len = Bx.input_dim

    x0 = m4.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi / 2, ub=np.pi / 2)

    m4.addConstr(x0[0] >= np.pi / 6)
    m4.addConstr(x0[0] <= np.pi / 2)

    m4.addConstr(x0[1] >= -np.pi / 2)
    m4.addConstr(x0[1] <= np.pi / 2)

    y_layer = MILP_Encode_NN(m4, x0, Bx)

    m4.setObjective(y_layer, GRB.MINIMIZE)

    # 优化
    m4.optimize()

    x0_result = solution_output(m4, 0)
    if len(x0_result) != 0:
        eye4 = False
    else:
        eye4 = True
    writeToCsv(filename, x0_result)


    return eye1 & eye2 & eye3 & eye4


# 初始区域的验证
def MILP_opt_initCon(Bx, Col):

    eye1 = False
    eye2 = False
    eye3 = False

    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # Xi 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi / 9, ub=np.pi / 9)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.pi / 2, ub=np.pi / 2)


    x2 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x2', lb=-np.pi / 2, ub=np.pi / 2)



    # k = 0

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx)
    m.setObjective(y_layer_for_x0, GRB.MAXIMIZE)

    m.optimize()

    violation_ex0 = solution_output(m, 1)
    if len(violation_ex0) != 0:
        eye1 = False
    else:
        eye1 = True


    # k = 1
    theNextPoint2(m, x0, x1, Col)
    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx)

    m.setObjective(y_layer_for_x1, GRB.MAXIMIZE)

    m.optimize()

    violation_ex1 = solution_output(m, 1)
    if len(violation_ex1) != 0:
        eye2 = False
    else:
        eye2 = True



    # k = 2
    theNextPoint2(m, x1, x2, Col)
    y_layer_for_x2 = MILP_Encode_NN(m, x2, Bx)

    m.setObjective(y_layer_for_x2, GRB.MAXIMIZE)

    m.optimize()

    violation_ex2 = solution_output(m, 1)
    if len(violation_ex2) != 0:
        eye3 = False
    else:
        eye3 = True


    filename = "Sampling/SamplingData/Xi_set_data.csv"
    writeToCsv(filename, violation_ex0)
    writeToCsv(filename, violation_ex1)
    writeToCsv(filename, violation_ex2)

    return eye1 & eye2 & eye3

    # debug

    #m.computeIIS()
    #m.write("model1.ilp")

# 蕴含条件的验证
def MILP_opt_Indicator(Bx, Col):

    eye = False
    m = gp.Model()

    # 配置
    m.setParam('Outputflag', 0)
    m.setParam('NonConvex', 2)
    # m.setParam('InfUnbdInfo', 1)

    x0_len = Bx.input_dim

    # X 区域约束
    x0 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x0', lb=-np.pi/2, ub=np.pi/2)

    x1 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x1', lb=-np.pi/2, ub=np.pi/2)
    theNextPoint2(m, x0, x1, Col, "Col(x0)")

    x2 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x2', lb=-np.pi/2, ub=np.pi/2)
    theNextPoint2(m, x1, x2, Col, "Col(x1)")

    x3 = m.addMVar((x0_len,), vtype=GRB.CONTINUOUS, name='x3', lb=-np.pi/2, ub=np.pi/2)
    theNextPoint2(m, x2, x3, Col, "Col(x2)")

    y_layer_for_x0 = MILP_Encode_NN(m, x0, Bx, "Bx(x0)")
    y_layer_for_x1 = MILP_Encode_NN(m, x1, Bx, "Bx(x1)")
    y_layer_for_x2 = MILP_Encode_NN(m, x2, Bx, "Bx(x2)")
    y_layer_for_x3 = MILP_Encode_NN(m, x3, Bx, "Bx(x3)")

    m.addConstr(y_layer_for_x0 <= 0)
    m.addConstr(y_layer_for_x1 <= 0)
    m.addConstr(y_layer_for_x2 <= 0)

    m.setObjective(y_layer_for_x3, GRB.MAXIMIZE)

    m.optimize()

    #print(m)
    violation_ex = solution_output(m, 2)
    if len(violation_ex) != 0:
        eye = False
    else:
        eye = True

    for v in m.getVars():
        print(f"{v.varName} == {v.x}")


    filename = "Sampling/SamplingData/X_set_data.csv"
    writeToCsv(filename, violation_ex)

    return eye

# ————————————————————————————————————————————————————————————————————————————————————————

def MILP_Encode_NN(m, x0, nn, str = "Bx__"):

    ''' MILP编码神经网络

    :param m: MILP Modle
    :param x0: NN 输入向量 （数学变量）
    :param nn: 需要编码的神经网络
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


# flag = 0: 求最小值 不安全区域条件
# flag = 1: 求最大值 初始条件
# flag = 2: 求最大值 蕴含条件

def solution_output(m, flag):
    x0 = []
    for v in m.getVars():
        if 'x0' in v.varName:
            x0.append(v.x)

    print("返回优化结果：")
    print(m.objVal)

    if flag == 0:
        if m.objVal > 0:
            print("不安全区域，条件验证满足，无反例")
            return []
        else:
            print("不安全区域，条件验证不满足，存在反例")
            print("反例：")
    elif flag == 1:
        if m.objVal <= 0:
            print("初始区域，条件验证满足，无反例")
            return []
        else:
            print("初始区域，条件验证不满足，存在反例")
            print("反例：")
    else:
        if m.objVal <= 0:
            print("蕴含条件验证满足，无反例")
            return []
        else:
            print("蕴含条件验证不满足，存在反例")
            print("反例：")
    x0 = np.array(x0).reshape((-1, 2))
    print(x0)
    return x0


def theNextPoint(m, x_pre, x_next, Col):
    alpha = 0.05
    u = MILP_Encode_NN(m, x_pre, Col)[0]

    # x0_dot = x_pre[1]
    # x1_dot = 9.8 * (x_pre[0] - x_pre[0]**3 / 6) + u

    x_pre_0_2 = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x_pre_0_3 = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    m.addConstr(x_pre_0_2 == x_pre[0] * x_pre[0])
    m.addConstr(x_pre_0_3 == x_pre_0_2 * x_pre[0])
    m.update()

    x0_dot = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)
    x1_dot = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)

    m.addConstr(x0_dot == x_pre[1])
    m.addConstr(x1_dot == 9.8 * (x_pre[0] - x_pre_0_3 / 6) + u)



    x0_dot_2 = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)
    x1_dot_2 = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)
    vec_mo_2 = m.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = np.inf)
    vec_mo = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=np.inf)

    m.addConstr(x0_dot_2 == x0_dot * x0_dot)
    m.addConstr(x1_dot_2 == x1_dot * x1_dot)
    m.addConstr(vec_mo_2 == x0_dot_2 + x1_dot_2 + 0.00001)

    m.addGenConstrPow(xvar = vec_mo_2, yvar = vec_mo, a = 0.5)


    vec_mo_neg_1 = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)

    m.addGenConstrPow(xvar = vec_mo_2, yvar = vec_mo, a = 0.5)
    m.addGenConstrPow(xvar=vec_mo, yvar=vec_mo_neg_1, a=-1)
    m.update()

    m.addConstr(x_next[0] == x_pre[0] + alpha * x0_dot * vec_mo_neg_1)
    m.addConstr(x_next[1] == x_pre[1] + alpha * x1_dot * vec_mo_neg_1)

    m.update()


def theNextPoint2(m, x_pre, x_next, Col, str = ''):

    alpha = 0.001

    u = MILP_Encode_NN(m, x_pre, Col, str)[0]

    # x0_dot = x_pre[1]
    # x1_dot = 9.8 * (x_pre[0] - x_pre[0]**3 / 6) + u

    x_pre_0_2 = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    x_pre_0_3 = m.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
    m.addConstr(x_pre_0_2 == x_pre[0] * x_pre[0])
    m.addConstr(x_pre_0_3 == x_pre_0_2 * x_pre[0])
    m.update()

    x0_dot = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)
    x1_dot = m.addVar(vtype = GRB.CONTINUOUS, lb = -np.inf, ub = np.inf)

    m.addConstr(x0_dot == x_pre[1])
    m.addConstr(x1_dot == 9.8 * (x_pre[0] - x_pre_0_3 / 6.0) + u)

    m.addConstr(x_next[0] == x_pre[0] + alpha * x0_dot)
    m.addConstr(x_next[1] == x_pre[1] + alpha * x1_dot)

    m.update()


def writeToCsv(filename, data):
    if len(data) == 0:
        return
    with open(filename, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        for i in range(data.shape[0]):
            csv_write.writerow(data[i, :])
            print(f"{filename},更新了一条记录")
