import torch
# pytorch device
device = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# Barrier certificates
############################################

# NN
Bx_INPUT_DIM = 2
Bx_OUTPUT_DIM = 1


############################################
# Controllers
############################################

# NN
Col_INPUT_DIM = 2
Col_OUTPUT_DIM = 1

# bound of controllers (optional)
Col_OUT_BOUND = 3

############################################
# parameters of k-ICBC
############################################

# X0  <= gama  and  implication condition <= gama  (gama <= 0)
GAMA = 0.0

# Xu  >= landa  (landa >= 0)
LANDA = 0.0

# $k$
K = 3

# x_i+1 = x_i + alpha * f(x_i, u_i)
alpha = 0.005

############################################
# f(x, u)
############################################


# expression
f_x = (
    "x2",
    "-10 * sin(x1) - 0.1 * x2 + u"
)

# variables
x_var_str = "x1,x2"
control_var_str = "u"
var_num = 3
x_var_num = 2
col_var_num = 1

# batch config
Xi_batch_size = 2 ** 7

Xu_batch_size = 2 ** 7

X_batch_size = 2 ** 7

# sample config
Xi_sample_len = 2 ** 6
Xu_sample_len = 2 ** 7
X_sample_len = 2 ** 7


# counterexamples sample config
tao = 0.05

# sampling among (x - tao, y - tao) -> (x + tao, y + tao)
mini_len = 2 ** 3
