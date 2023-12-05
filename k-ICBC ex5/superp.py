import torch
# pytorch device
device = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# Barrier certificates
############################################

# NN
Bx_INPUT_DIM = 4
Bx_OUTPUT_DIM = 1


############################################
# Controllers
############################################

# NN
Col_INPUT_DIM = 4
Col_OUTPUT_DIM = 1


############################################
# parameters of k-ICBC
############################################

# X0  <= gama  and  implication condition <= gama  (gama <= 0)
GAMA = 0.0

# Xu  >= landa  (landa >= 0)
LANDA = 0.0

# $k$
K = 2

# x_i+1 = x_i + alpha * f(x_i, u_i)
alpha = 0.005

############################################
# f(x, u)
############################################


# expression
f_x = (
    "-x1 - x4 + u",
    "x1 - x2 + x1 ** 2 + u",
    "-x3 + x4 + x2 ** 2",
    "x1 - x2 - x4 + x3 ** 3 - x4 ** 3"
)

# variables
x_var_str = "x1,x2,x3,x4"
control_var_str = "u"
var_num = 5
x_var_num = 4
col_var_num = 1

# batch config
Xi_batch_size = 2 ** 7

Xu_batch_size = 2 ** 7

X_batch_size = 2 ** 7

# sample config
Xi_sample_len = 2 ** 3
Xu_sample_len = 2 ** 4
X_sample_len = 2 ** 4


# counterexamples sample config
tao = 0.05

# sampling among (x - tao, y - tao) -> (x + tao, y + tao)
mini_len = 2 ** 3