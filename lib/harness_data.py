import os
import torch
import numpy as np
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from harness_options import Options


# 0 : Defect
# 1 : Glue
def load_data(opt):
    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': False}

    transform = transforms.Compose([
        transforms.Resize(opt.isize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = {}
    dataset['train'] = ImageFolder(root='../../data/harness_paper_dataset/sample/train/', transform=transform)
    dataset['test'] = ImageFolder(root='../../data/harness_paper_dataset/sample/test/', transform=transform)

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)))
                  for x in splits}

    return dataloader