from __future__ import print_function

from options import Options
from lib.data import load_data

def train():
    opt = Options().parse()
    dataloader = load_data(opt)


if __name__ == '__main__':
    train()