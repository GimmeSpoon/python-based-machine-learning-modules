from abc import abstractmethod, ABC
from typing import Callable, Union

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from base import Atts

__all__ = [
    "TorchParameters",
    "TorchModel",
]

class TorchParameters(Atts):
    def __init__(self, learning_rate:float, dict: dict, **kwargs) -> None:
        super().__init__(dict, kwargs)
        self.lr = learning_rate

class Model(ABC):
    r"""Base class for model
        has train and infer methods
    """
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def state(self):
        pass

class TorchModel(Model):
    r"""A class for Pytorch model(nn.Module) and other essential components such as optimizer or criterion(loss functions)
    
    You can train and infer with the instance of this class, but it does not contain any data loading features.
    This class is intended to group essential parts of pytorch learning and be registered to Trainer instances to load data needed.
    But on your favor, you can use this class solely without Trainer. Just make sure to process data properly.
    After inferring, it will save predicted result and target data separtely, but does not provide any information such as evaluation.
    """

    def __init__(
            self,
            net:nn.Module,
            hp:TorchParameters,
            cr:torch.nn.Module,
            opt:torch.optim.Optimizer,
            sch:torch.optim.lr_scheduler._LRScheduler,
            dev:Union[int, str, torch.device]
        ) -> None:
        super().__init__()

        if not hp.validate_essentials():
            raise AttributeError("Hyperparameters are not valid.")

        #basic constitues
        self.net = net
        self.eval = eval
        self.hp = hp
        self.criterion = cr
        self.opt = opt
        self.sch = sch

        #Environment
        self.set_device(dev)

        #Link
        self.trainer = None

    def set_device(self, dev:Union[int, str, torch.device], place:bool=True) -> None:
        if isinstance(dev, int) or isinstance(dev, str):
            self.dev = torch.device(dev)
        else:
            self.dev = dev
        if place:
            self.net.to(self.dev)

    def distributed(fn:Callable) -> Callable:
        def _dist(self:TorchModel, *args, **kwargs):
            self.net = DDP(self.net)
            return fn(self, args, kwargs)
        return _dist
          
    def train(self, x:Tensor, y:Tensor) -> Tensor:
        pred = self.net(x)
        loss:Tensor = self.criterion(pred, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss

    def test(self, x:Tensor, y:Tensor) -> tuple:
        output = self.net(x)
        result = (output, y)
        return result

    def infer(self, x:Tensor) -> Tensor:
        return self.net

    def state(self)->dict:
        return {
            'net' : self.net.state_dict(),
            'opt' : self.opt.state_dict(),
            'sch' : self.sch.state_dict() if self.sch is not None else None,
            'hp' : vars(self.hp)
        }

    def save(self, epoch:int, loss:Tensor, fname:str='checkpoint.pt'):
        torch.save({
            'net' : self.net.state_dict(),
            'opt' : self.opt.state_dict(),
            'sch' : self.sch.state_dict() if self.sch is not None else None,
            'hp' : vars(self.hp)
        }, fname)

    def load(self, path):
        with torch.load(path) as checkpoint:
            self.net.load_state_dict(checkpoint['net'])
            self.opt.load_state_dict(checkpoint['opt'])
            self.sch.load_state_dict(checkpoint['sch'])
            self.hp.__dict__.update( checkpoint['hp'] )

    