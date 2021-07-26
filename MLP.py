import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims=[784, 256, 64, 64, 62], bn=False, bias=True):
        super(MLP, self).__init__()
        self.net = nn.ModuleList([])
        for i in range(0, len(dims)-2):
            self.net.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            if bn:
                self.net.append(nn.BatchNorm1d(dims[i+1]))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Linear(dims[-2], dims[-1], bias=bias))

    def forward(self, x):
        #x = x.flatten(start_dim=-3, end_dim=-1)
        for layer in self.net:
            #print("bruh")
            x = layer(x)
        return x

    def normalize(self):
        return None

    

class scalable_linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super(scalable_linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias)
        self.scaling = nn.Parameter(torch.ones(dim_out))

    def forward(self, x):
        normlist = torch.linalg.norm(self.linear.weight, dim=-1)
        return self.linear(x)* self.scaling/normlist

    def normalize(self, old_norms=None):
        #print(self.linear.weight)
        with torch.no_grad():
            if old_norms is not None: # TODO: Passt das mit den Dimensionen?
                self.linear.weight.mul_(old_norms.transpose(-2,-1)) # = torch.nn.Parameter((self.weights.T*preweights).T, requires_grad=True)
            normlist = torch.linalg.norm(self.linear.weight, keepdim=True, dim=-1)
            #normlist = torch.sum(torch.abs(self.linear.weight.data)**p, dim=-1).unsqueeze(-1)**(1/p)
            self.linear.weight.data.mul_(1/normlist)
            #self.linear.bias = torch.nn.Parameter(self.bias*normlist, requires_grad=True)
            normlist*=self.scaling.unsqueeze(-1)
            self.scaling.data.mul_(1/self.scaling)
        return normlist



class MLP_normed(nn.Module):
    def __init__(self, dims=[784, 256, 64, 64, 62], bias=True):
        super().__init__()
        self.layers = nn.ModuleList([scalable_linear(dims[i], dims[i+1], bias=bias) for i in range(len(dims)-1)])
        self.outnorms = nn.Parameter(torch.ones(dims[-1]), requires_grad=True)

    def forward(self, x):
        for layer in self.layers:
            normlist = torch.linalg.norm(layer.linear.weight, dim=-1)
            x = F.relu(layer(x)) * 1/normlist
        return self.outnorms*x

    def normalize(self):
        old_norms = None
        returnnorms = torch.zeros((4, 2))
        for i, layer in enumerate(self.layers):
            old_norms = layer.normalize(old_norms)
            returnnorms[i,0] = torch.max(old_norms)
            returnnorms[i,1] = torch.min(old_norms)
        with torch.no_grad():
            self.outnorms.data.mul_(old_norms.squeeze())
        return returnnorms



class Trainer(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        x = x.squeeze().flatten(start_dim=-2, end_dim=-1)
        x -= x.mean(keepdim=True, dim=-1)
        x *= 1/x.std(keepdim=True, dim=-1)
        #print(x.shape)
        return self.network(x)

    def normalize(self):
        returnnorms = self.network.normalize()
        return returnnorms


def NetworkGenerator():
    yield MLP_normed(bias=True)
    yield MLP(bias=True)
    yield MLP(bn=True, bias=True)