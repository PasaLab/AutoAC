import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from utils.tools import *

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect_Nasp(object):
    
    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.network_momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def step(self, X, y, eta=None, network_optimizer=None, unrolled=False):
        self.optimizer.zero_grad()
        self._backward_step(X, y, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, X, y, is_valid=True):
        self.model.binarization()
        loss = self.model._loss(X, y, is_valid)
        loss.backward()
        self.model.restore()
        