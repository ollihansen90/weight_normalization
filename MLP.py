import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import sqrt

class MLP(nn.Module):
    def __init__(self, dims=[784, 256, 64, 64, 62], bn=False, bias=True, centrify=False):
        super(MLP, self).__init__()
        self.net = nn.ModuleList([])
        for i in range(0, len(dims)-2):
            self.net.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            if bn:
                self.net.append(nn.BatchNorm1d(dims[i+1]))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Linear(dims[-2], dims[-1], bias=bias))
        self.centrify = centrify

    def forward(self, x):
        #x = x.flatten(start_dim=-3, end_dim=-1)
        for layer in self.net:
            with torch.no_grad():
                if self.centrify:
                    x.add(-x.mean(dim=-1).unsqueeze(-1))
            #print("bruh")
            x = layer(x)
        return x

    def normalize(self):
        return None

    

class scalable_linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, bias_init_norm=True, start_norm=sqrt(3), own_norm=1/sqrt(3)):
        super(scalable_linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.own_norm = own_norm
        """with torch.no_grad():
            normlist = 1/start_norm*torch.linalg.norm(self.linear.weight.data, keepdim=True, dim=-1)+1e-10
            self.linear.weight.data.mul_(1/normlist)
            #print(torch.mean(normlist).item())
            if bias_init_norm:
                self.linear.bias.data.mul_(1/normlist.squeeze())
"""
    def forward(self, x):
        #normlist = torch.linalg.norm(self.linear.weight.data, dim=-1)
        return self.linear(x)#* self.scaling/normlist

    def normalize(self, target_norm=None, old_norms=None):
        if target_norm is None:
            target_norm = self.own_norm
        #print(self.linear.weight)
        with torch.no_grad():
            if old_norms is not None:
                self.linear.weight.mul_(old_norms.transpose(-2,-1)) # = torch.nn.Parameter((self.weights.T*preweights).T, requires_grad=True)
                #normlist = torch.sum(torch.abs(self.linear.weight.data)**p, dim=-1).unsqueeze(-1)**(1/p)
            normlist = 1/target_norm*torch.linalg.norm(self.linear.weight.data, keepdim=True, dim=-1)+1e-10
            self.linear.weight.data.mul_(1/normlist)
            self.linear.bias.data.mul_(1/normlist.squeeze())
        return normlist



class scalable_linear_noBubble(nn.Module):
    # Ähnlich wie scalable_linear, nur dass hier nicht geblubbert wird.
    def __init__(self, dim_in, dim_out, bias=True, start_norm=sqrt(3), own_norm=1/sqrt(3)):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.own_norm = own_norm
        with torch.no_grad():
            normlist = 1/start_norm*torch.linalg.norm(self.linear.weight.data, keepdim=True, dim=-1)+1e-10
            self.linear.weight.data.mul_(1/normlist)
            self.linear.bias.data.mul_(1/normlist.squeeze())
        self.scales = nn.Parameter(torch.ones(dim_out), requires_grad=False)

    def forward(self, x):
        return self.scales*self.linear(x)#* self.scaling/normlist

    def normalize(self, old_norm=None):
        #print(self.linear.weight)
        with torch.no_grad():
            normlist = 1/self.own_norm*torch.linalg.norm(self.linear.weight.data, keepdim=True, dim=-1)+1e-10
            self.linear.weight.data.mul_(1/normlist)
            self.linear.bias.data.mul_(1/normlist.squeeze())
            self.scales.mul_(normlist.squeeze())
        return None


class MLP_normed(nn.Module):
    def __init__(   self, 
                    dims=[784, 256, 64, 64, 62], 
                    bias=True, 
                    bias_init_norm=False,
                    own_norm=[sqrt(3)]+3*[sqrt(2)], # Erstes Layer bekommt Norm sqrt(3), da vorher kein ReLU
                    outnorm_init=1, # braucht man vermutlich nicht
                    bubble=True
                ):
        super().__init__()
        self.bubble = bubble
        self.depth = len(dims)
        self.own_norm = own_norm
        if bubble:
            self.layers = nn.ModuleList([
                            scalable_linear(
                                dims[i], dims[i+1], bias=bias, 
                                bias_init_norm=bias_init_norm, 
                                start_norm=own_norm[i], # alt, müsste mal beseitigt werden
                                own_norm=own_norm[i]
                            ) for i in range(len(dims)-1)
                            ])
        else:
            self.layers = nn.ModuleList([
                            scalable_linear_noBubble(
                                dims[i], dims[i+1], bias=bias, 
                                start_norm=own_norm[i]
                            ) for i in range(len(dims)-1)])

        self.outnorms = nn.Parameter(outnorm_init*torch.ones(dims[-1]), requires_grad=False)
        with torch.no_grad():
            self.normalize(own_norm, bubble=True)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx<(self.depth-2):
                x = F.relu(x)# * 1/normlist
        return self.outnorms*x

    def normalize(self, target_norm=None, bubble=True):
        if target_norm is None:
            target_norm = self.own_norm
        old_norms = None
        #returnnorms = torch.zeros((4, 2))
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                old_norms = layer.normalize(target_norm=target_norm[i], old_norms=old_norms)
                if not bubble:
                    old_norms = None
        if bubble:
            self.outnorms.data.mul_(old_norms.squeeze())

        return None #"""returnnorms"""

class MLP_normed_alt(nn.Module):
    def __init__(self, dims=[784, 256, 64, 64, 62], bias=True, bias_init_norm=True, centrify=False, start_norm=sqrt(3), own_norm=1/sqrt(3), outnorm_init=1, bubble=True):
        super().__init__()
        self.bubble = bubble
        self.depth = len(dims)
        self.own_norm = own_norm
        #print(self.depth)
        if bubble:
            self.layers = nn.ModuleList([
                            scalable_linear(
                                dims[i], dims[i+1], bias=bias, 
                                bias_init_norm=bias_init_norm, 
                                start_norm=start_norm, 
                                own_norm=own_norm) 
                                for i in range(len(dims)-1)
                            ])
        else:
            self.layers = nn.ModuleList([scalable_linear_noBubble(dims[i], dims[i+1], bias=bias, start_norm=start_norm) for i in range(len(dims)-1)])
        self.outnorms = nn.Parameter(outnorm_init*torch.ones(dims[-1]), requires_grad=False)
        self.centrify = centrify
        with torch.no_grad():
            self.normalize(start_norm, bubble=True)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            with torch.no_grad():
                if self.centrify:
                    x.add(-x.mean(dim=-1).unsqueeze(-1))
            #normlist = torch.linalg.norm(layer.linear.weight, dim=-1)
            if idx<(self.depth-1):
                #print(idx, self.depth)
                x = F.relu(x)# * 1/normlist
        return self.outnorms*x

    def normalize(self, target_norm=None, bubble=True):
        if target_norm is None:
            target_norm = self.own_norm
        old_norms = None
        #returnnorms = torch.zeros((4, 2))
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                old_norms = layer.normalize(target_norm=target_norm, old_norms=old_norms)
                if not bubble:
                    old_norms = None
            """if bubble:
                returnnorms[i,0] = torch.max(old_norms)
                returnnorms[i,1] = torch.min(old_norms)"""
        #with torch.no_grad():
        #print(old_norms)
        if bubble:
            self.outnorms.data.mul_(old_norms.squeeze())
        #print(self.outnorms)
        return None #"""returnnorms"""


class Normalizer():
    def __init__(self, device, load=True):
        if load:
            self.img_mu = torch.load("normalizer_data/img_mu.pt")
            self.img_std = torch.load("normalizer_data/img_std.pt")
        else:
            self.img_mu = torch.zeros([28**2]).to(device)
            self.img_std = torch.zeros([28**2]).to(device)
            self.N = 0

    def __call__(self, imgtensor):
        return (imgtensor-self.img_mu)/(self.img_std+1e-10)

    def estimate(self, imgtensor):
        self.N += imgtensor.shape[0]
        self.img_mu += torch.mean(imgtensor, dim=0).squeeze()
        self.img_std += torch.mean(imgtensor**2, dim=0).squeeze()

    def estimate_done(self):
        self.img_mu *= 1/self.N
        self.img_std = torch.sqrt(self.img_std/self.N-self.img_mu**2)

    def plot_estimates(self):
        plt.figure()
        plt.imshow(self.img_mu.reshape([28,28]))
        plt.savefig("img_mu.png")
        plt.figure()
        plt.imshow(self.img_std.reshape([28,28]))
        plt.savefig("img_std.png")

    def save_imgs(self):
        torch.save(self.img_mu, "normalizer_data/img_mu.pt")
        torch.save(self.img_std, "normalizer_data/img_std.pt")


class Trainer(nn.Module):
    def __init__(self, network, normalizer):
        super().__init__()
        self.network = network
        self.normalizer = normalizer

    def forward(self, x):
        with torch.no_grad():
            x = x.squeeze().flatten(start_dim=-2, end_dim=-1)
            x = self.normalizer(x)
            """x -= x.mean(keepdim=True, dim=-1)
            x *= 1/x.std(keepdim=True, dim=-1)"""
        #print(x.shape)
        return self.network(x)

    def normalize(self):
        returnnorms = self.network.normalize()
        return returnnorms


def NetworkGenerator(n):
    #yield MLP_normed(bias=True, bias_init_norm=True, centrify=False, bubble=True, start_norm=1/sqrt(3), own_norm=1/sqrt(3), outnorm_init=1)
    ##yield MLP_normed(bias=True, centrify=False, bubble=True, start_norm=sqrt(3), outnorm_init=1)
    #yield MLP(bn=True)
    for _ in range(n):
        yield MLP_normed(bias_init_norm=False, own_norm=[sqrt(3)]+3*[sqrt(2)])
    for _ in range(n):
        yield MLP_normed(bias_init_norm=True, own_norm=[1/sqrt(3)]+3*[1/sqrt(2)])
    for _ in range(n):
        yield MLP(bn=False)
    """for _ in range(n):
        yield MLP_normed(bias=True, bias_init_norm=False, centrify=False, bubble=True, start_norm=sqrt(3), own_norm=1/sqrt(3), outnorm_init=1)
    for _ in range(n):
        yield MLP(bn=True)"""