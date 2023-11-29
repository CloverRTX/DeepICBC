import torch


def torch_add(m, n):
    return m + n


def torch_sub(m, n):
    return m - n


def torch_mul(m, n):
    return m * n


def torch_power(m, n):
    return m ** n

def torch_sin(m):
    return torch.sin(m)



operator_dict = {
    '+': torch_add,
    '-': torch_sub,
    '*': torch_mul,
    '**': torch_power,
    'sin': torch_sin
}
