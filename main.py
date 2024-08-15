import torch


if __name__ == '__main__':
    x = torch.rand(2, 640)
    x = x.requires_grad_(True)
    a = x.float()
    b = a
    c = (a + b)[0][0]

    print(torch.autograd.grad(c.requires_grad_(True), [x])[0])
