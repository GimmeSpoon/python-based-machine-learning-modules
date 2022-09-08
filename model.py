from abc import abstractmethod, ABC
from typing import Callable, Literal

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
            sch:torch.optim.lr_scheduler._LRScheduler
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

        #Training result
        self.train_s_hook = [] # step_hook
        self.train_e_hook = [] # epoch_hook
        self.test_hook = []
        self.infer_hook = []   # infer_hook
        self.losses = []
        self.trained = False
        self.pred = Tensor()
        self.target = Tensor()

        #Link
        self.trainer = None

    def distributed(fn:Callable) -> Callable:
        def _dist(self:TorchModel, *args, **kwargs):
            self.net = DDP(self.net)
            return fn(self, args, kwargs)
        return _dist
    
    def train_step(self, x:Tensor, y:Tensor, device:torch.device) -> Tensor:
        x, y = x.to(device), y.to(device)
        pred = self.net(x)
        loss:Tensor = self.criterion(pred, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.__train_step_hook(x, y, pred, loss, self.hp)
        return loss

    def add_hooker(
            self,
            time:Literal['s', 'e', 't', 'i'],
            hook:Callable
        ) -> None:
        if time == 's':
            self.train_s_hook.append(hook)
        elif time == 'e':
            self.train_e_hook.append(hook)
        elif time == 't':
            self.test_hook.append(hook)
        elif time == 'i':
            self.infer_hook.append(hook)

    def __train_step_hook(self, x:Tensor, y:Tensor, pred:Tensor, loss:Tensor, hp:TorchParameters) -> None:
        for h in self.__train_s_hook:
            h(x, y, pred, loss, hp)

    def __train_epoch_hook(self, y:Tensor, pred:Tensor, loss:Tensor, hp:TorchParameters) -> None:
        for h in self.__train_e_hook:
            h(x, y, pred, loss, hp)

    def train(self, loader:DataLoader, device:torch.device, epoch:int=0, loss:float=0., bar:bool=True):
        last_epoch = epoch
        self.net.to(device)
        self.net.train()
        for epoch in tqdm(range(epoch, self.hp.epochs), desc='Total', unit='epoch', position=1, leave=False, disable=not bar):
            loss = 0.0
            for i, (x, y) in enumerate(tqdm(loader, desc='Batch', postfix={'Loss':'%.5F'%(loss / loader.batch_size)}, leave=False, disable=not bar)):
                curloss = self.train_step(x, y, device)
                loss += curloss.item() * len(y) / len(loader)
            #Log
            self.losses.append(loss)
            #Save
            if epoch % self.checkpoint == self.checkpoint - 1:
                last_epoch = epoch
                self.save(epoch+1, self.net.state_dict(), loss, f'{self.net.__class__.__name__}_{epoch+1}.pt')
            self.sch.step()
            self.__train_epoch_hook()

        self.trained = True
        
        #save after training
        if last_epoch != epoch:
            self.save(epoch+1, self.net.state_dict(), loss, f'{self.net.__class__.__name__}_{epoch+1}.pt')

    def test(self, loader:DataLoader, device:torch.device, bar:bool=True)->tuple:
        self.net.eval()
        self.net.to(device)
        with torch.no_grad():
            for x, y in enumerate(tqdm(loader, disable=not bar)):
                x, y = x.to(device), y.to(device)
                output = self.net(x)
                self.pred = torch.cat([self.pred, output])
                self.target = torch.cat([self.eval, y])
        result = (self.pred, self.target)
        return result

    def infer(self, loader:DataLoader, device:torch.device, bar:bool=True) -> Tensor:
        self.net.eval()
        self.net.to(device)
        output = []
        with torch.no_grad():
            for x in enumerate(tqdm(loader, disable=not bar)):
                x = x.to(device)
                output.append(self.net(x))
        return output

    def __state__(self)->dict:
        return {
            'parm' : self.net.state_dict(),
            'opt' : self.opt.state_dict(),
            'sch' : self.sch.state_dict() if self.sch is not None else None,
            'hp' : vars(self.hp)
        }

    def save(self, epoch:int, loss:Tensor, fname:str='checkpoint.pt'):
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            'sch': self.sch.state_dict()
        }, fname)

    def load(self, path):
        with torch.load(path) as checkpoint:
            self.net.load_state_dict(checkpoint['parm'])
            self.opt.load_state_dict(checkpoint['opt'])
            self.sch.load_state_dict(checkpoint['sch'])
            return (checkpoint['epoch'], checkpoint['loss'])

    