from torch import nn

class NN_Barrier(nn.Module):


    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(NN_Barrier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim)
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