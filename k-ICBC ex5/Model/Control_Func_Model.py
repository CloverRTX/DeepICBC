import torch
from torch import nn

class NN_Control(nn.Module):


    def __init__(self, input_dim, output_dim):
        super(NN_Control, self).__init__()

        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, output_dim)
            #nn.Hardtanh(-out_bound, out_bound)
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze(1)

    def serveForVerify(self):
        W_b_list = []
        for i in self.state_dict():
            var = self.state_dict()[i].cpu().detach().numpy()
            W_b_list.append(var)
        return W_b_list
