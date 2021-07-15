from MLP import MLP_normed, scalable_linear
import torch

layer = scalable_linear(3,8)
layer.normalize()

normedMLP = MLP_normed()
normedMLP.normalize()
print("Start")

for name, param in normedMLP.named_parameters():
    print(name)
    print(torch.linalg.norm(param, dim=-1))


#print(torch.tensor([0,1,-10])==False)

"""print()
test = 0.1*torch.randn(10)+1
print(test)
test = torch.sigmoid(test)
print(test)
test = torch.logit(test)
print(test)"""