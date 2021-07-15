import torch
from torch import nn
import torch.nn.functional as F

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

class scalable_linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super(scalable_linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(dim_out, dim_in), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(dim_out), requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weights, bias=self.bias)
    
    def normalize_weights(self, preweights=None):
        if preweights: # TODO: Passt das mit den Dimensionen?
            with torch.no_grad():
                self.weights = torch.nn.Parameter((self.weights.T*preweights).T, requires_grad=True)
        normlist = torch.linalg.norm(self.weights, dim=-1)
        print(self.weights)
        print(self.bias)
        print(normlist)
        with torch.no_grad():
            self.weights = torch.nn.Parameter((self.weights.T/normlist).T, requires_grad=True)
            self.bias = torch.nn.Parameter(self.bias*normlist, requires_grad=True)
        print(self.weights)
        print(self.bias)
        print(torch.linalg.norm(self.weights, dim=-1))
