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

def torch_cos(m):
    return torch.cos(m)

def torch_tan(m):
    return torch.tan(m)


operator_dict = {
    '+': torch_add,
    '-': torch_sub,
    '*': torch_mul,
    '**': torch_power,
    'sin': torch_sin,
    'cos': torch_cos,
    'tan': torch_tan
}
