import torch

def saveNN(model, pathname):
    torch.save(model, pathname)

def loadNN(pathname):
    return torch.load(pathname)