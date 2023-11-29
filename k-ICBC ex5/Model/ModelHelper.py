import torch

def saveNN(model, pathname):
    '''保存训练的神经网络模型

    :param model:要保存的模型
    :param pathname:保存的文件路径
    :return: None
    '''
    torch.save(model, pathname)

def loadNN(pathname):
    '''读取保存的NN Model

    :param pathname:模型的保存路径
    :return: NN model
    '''
    return torch.load(pathname)