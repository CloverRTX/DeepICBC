from torch import nn

class NN_Barrier(nn.Module):


    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim
        super(NN_Barrier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
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

    '''
    def setWeight(self, key=None, new_value=None):
        for k, v in self.state_dict().items():
            if k == key:
                self.state_dict()[key].copy_(new_value)

    
    def getNNEachLayerReturn(self, point):
        y1 = self.layers[0](point)
        z1 = self.layers[1](y1)
        y2 = self.layers[2](z1)
        z2 = self.layers[3](y2)
        y3 = self.layers[4](z2)
        print("——————————NN_Layers_Info——————————")
        print(f"Y1 = {y1}")
        print(f"Z1 = {z1}")
        print(f"Y2 = {y2}")
        print(f"Z2 = {z2}")
        print(f"Y3 = {y3}")
    '''