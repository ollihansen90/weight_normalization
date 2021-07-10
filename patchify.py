import torch
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime as dt

def patchify(img, n_patches=16):
    """img = torch.tensor_split(img, n_patches_x, dim=-1)
    img = torch.stack(img, dim=0).flatten(start_dim=0, end_dim=1)
    img = torch.tensor_split(img, n_patches_y, dim=-2)
    img = torch.stack(img, dim=0).flatten(start_dim=0, end_dim=1)"""
    img = torch.stack(torch.chunk(img, int(sqrt(n_patches)), dim=-1), dim=-3)
    #print(img.shape)
    img = torch.cat(torch.chunk(img, int(sqrt(n_patches)), dim=-2), dim=-3)
    img = img.flatten(start_dim=-2, end_dim=-1)
    #print("hier", img.shape)
    #img = img.flatten(start_dim=-4, end_dim=-3)
    return img.squeeze()

def buildgrid():
    n = 4
    grid = torch.tensor(list(range(n)))
    grid = grid.unsqueeze(0)
    grid = grid.expand(n,n)
    grid = grid + n*torch.tensor(list(range(n))).unsqueeze(1)
    return grid.t().flatten()
