from MLP import scalable_linear
import torch

layer = scalable_linear(2,1)
layer.normalize_weights()

print()
test = 0.1*torch.randn(10)+1
print(test)
test = torch.sigmoid(test)
print(test)
test = torch.logit(test)
print(test)