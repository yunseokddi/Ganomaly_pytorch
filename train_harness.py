from __future__ import print_function

from harness_options import Options
from lib.model import Ganomaly
from lib.harness_data import load_data


def train():
    opt = Options().parse()
    dataloader = load_data(opt)
    model = Ganomaly(opt, dataloader)

    model.train()


if __name__ == '__main__':
    train()
