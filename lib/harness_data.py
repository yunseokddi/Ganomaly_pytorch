import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def load_data(opt):
    transform = transforms.Compose([
        transforms.Resize(opt.isize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    classes = {
        'glue': 0, 'defect': 1
    }

    dataset = {}
    dataset['train'] = ImageFolder(root='./data/harness', transform=transform)
    dataset['test'] = ImageFolder(root='./data/harness', transform=transform)


