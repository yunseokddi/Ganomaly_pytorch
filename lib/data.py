import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def load_data(opt):
    if opt.dataroot == '':
        opt.dataroot = './data{}'.format(opt.dataset)

    splits = ['train', 'test']
    drop_last_batch = {'train': True, 'test': False}
    shuffle = {'train': True, 'test': False}

    transform = transforms.Compose(
        [
            transforms.Resize(opt.isize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    classes = {
        'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
        'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    }

    dataset = {}
    dataset['train'] = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset['test'] = CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataset['train'].data, dataset['train'].targets, \
    dataset['test'].data, dataset['test'].targets = get_cifar_anomaly_dataset(
        trn_img=dataset['train'].data,
        trn_lbl=dataset['train'].targets,
        tst_img=dataset['test'].data,
        tst_lbl=dataset['test'].targets,
        abn_cls_idx=classes[opt.abnormal_class],
        manualseed=opt.manualseed
    )

    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                 batch_size=opt.batchsize,
                                                 shuffle=shuffle[x],
                                                 num_workers=int(opt.workers),
                                                 drop_last=drop_last_batch[x],
                                                 worker_init_fn=(None if opt.manualseed == -1
                                                                 else lambda x: np.random.seed(opt.manualseed)))
                  for x in splits}

    return dataloader


def get_cifar_anomaly_dataset(trn_img, trn_lbl, tst_img, tst_lbl, abn_cls_idx=0, manualseed=-1):
    """[summary]

       Arguments:
           trn_img {np.array} -- Training images
           trn_lbl {np.array} -- Training labels
           tst_img {np.array} -- Test     images
           tst_lbl {np.array} -- Test     labels

       Keyword Arguments:
           abn_cls_idx {int} -- Anomalous class index (default: {0})

       Returns:
           [np.array] -- New training-test images and labels.
       """
    trn_lbl = np.array(trn_lbl)
    tst_lbl = np.array(tst_lbl)

    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]  # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal testing images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal testing images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal testing labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal testing labels.

    # 0: Normal, 1: Abnormal
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_img[:] = 1

    if manualseed != -1:
        # Random seed.
        # Concatenate the original train and test sets.
        nrm_img = np.concatenate((nrm_trn_img, nrm_tst_img), axis=0)
        nrm_lbl = np.concatenate((nrm_trn_lbl, nrm_tst_lbl), axis=0)
        abn_img = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
        abn_lbl = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

        # Split the normal data into the new train and tests.
        idx = np.arange(len(nrm_lbl))
        np.random.seed(manualseed)
        np.random.shuffle(idx)

        nrm_trn_len = int(len(idx) * 0.80)
        nrm_trn_idx = idx[:nrm_trn_len]
        nrm_tst_idx = idx[nrm_trn_len:]

        nrm_trn_img = nrm_img[nrm_trn_idx]
        nrm_trn_lbl = nrm_lbl[nrm_trn_idx]
        nrm_tst_img = nrm_img[nrm_tst_idx]
        nrm_tst_lbl = nrm_lbl[nrm_tst_idx]

    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.copy(nrm_tst_img)
    new_tst_lbl = np.copy(nrm_tst_lbl)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl
