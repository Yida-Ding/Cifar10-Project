import torch
import json

# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3   # element-wise product
# out = z.mean()

# res = torch.autograd.grad(outputs=out, inputs=x)
# print(res)


# x = torch.ones(2)  # input tensor
# y = torch.zeros(3)  # expected output
# W = torch.randn(2, 3, requires_grad=True) # weights
# b = torch.randn(3, requires_grad=True) # bias vector
# z = torch.matmul(x, W)+b # output
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# loss.backward()
# print(W.grad) #OK
# print(b.grad) #OK
# print(x.grad)


