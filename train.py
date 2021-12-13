from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly


def train():
    opt = Options().parse()
    dataloader = load_data(opt)
    model = Ganomaly(opt, dataloader)

    model.train()


if __name__ == '__main__':
    train()
