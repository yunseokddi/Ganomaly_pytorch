import torch

def l1_loss(input, target):
    return torch.mean(torch.abs(input - target))

def l2_loss(input, target, size_average=True):
    if size_average:
        return torch.mean(torch.pow((input -target), 2))
    else:
        return torch.pow((input - target), 2)