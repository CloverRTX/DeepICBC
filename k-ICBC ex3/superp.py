import torch
# pytorch 设备选取
device = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# 障碍函数
############################################

# NN 输入
Bx_INPUT_DIM = 3
# NN 输出
Bx_OUTPUT_DIM = 1


############################################
# 控制器
############################################

# NN 输入
Col_INPUT_DIM = 3
# NN 输出
Col_OUTPUT_DIM = 1


############################################
# k - 归纳障碍函数  超参数设置
############################################

# 初始区域  <= gama  and  蕴含条件 <= gama  (gama 必须小于等于 0)
GAMA = 0.0

# 不安全区域  >= landa  (landa 必须大于等于 0)
LANDA = 0.0

# k步设置  k次迭代
K = 1

# 步长设置 要求 下一个点 与 当前点 的距离
# step_len = 0.05

# 固定alpha的值
# x_i+1 = x_i + alpha * f(x)
alpha = 0.001

############################################
# 动力系统 向量场
############################################


# 原系统
f_x_ori = (
    "x2",
    "30 * sin(x1) + 100 * cos(x1) * tan(x3) + 15 * cos(x1) / (cos(x3) ** 2) * u",
    "u"
)
# 转换后的系统
# 转换公式 u = u'cos^2(x3) - 20cos^(x3)tan(x3)
f_x = (
    "x2",
    "30 * sin(x1) + 15 * u * cos(x1)",
    "u * (cos(x3) ** 2) - 20 * (cos(x3) ** 2) * tan(x3)"
)


# 变量串
x_var_str = "x1,x2,x3"
control_var_str = "u"
var_num = 4
x_var_num = 3
col_var_num = 1

# batch 设置
Xi_batch_size = 2 ** 8

Xu_batch_size = 2 ** 8

X_batch_size = 2 ** 8

# 采样长度 (每个维度)
Xi_sample_len = 2 ** 5
Xu_sample_len = 2 ** 6
X_sample_len = 2 ** 6


# 反例周围采样指标
tao = 0.05

# 采样长度
mini_len = 2 ** 3
# (x - tao, y - tao) -> (x + tao, y + tao) 之间采样