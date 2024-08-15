import torch


if __name__ == '__main__':
    x = torch.rand(2, 640)
    x = x.requires_grad_(True)
    print(f'x:{x.requires_grad}')
    a = torch.cat([x] * 2)
    print(f'a:{a.requires_grad}')
    print(f'a.is_leaf:{a.is_leaf}')
    d = torch.tensor(10.)

    print(torch.autograd.grad(d.requires_grad_(True), [x], allow_unused=True)[0])
    print(torch.cuda.device_count())
