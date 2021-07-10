from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim=28**2, hidden_dims=3*[64], num_classes=62):
        super(MLP, self).__init__()
        self.net = nn.ModuleList([nn.Linear(in_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)-1):
            self.net.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_dims[-1], num_classes))

    def forward(self, x):
        x = x.flatten(start_dim=-3, end_dim=-1)

        for layer in self.net:
            x = layer(x)

        return x

    def normalize(self):
        pass