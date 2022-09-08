import os
import multiprocessing as mp
from typing import Callable, Final, Iterable, Sequence, Tuple, TypeVar, Union
from abc import abstractmethod, ABC

from numpy import ndarray
from pandas import DataFrame
import paramiko
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import TorchModel, TorchParameters
from remote import ClientInfo
from base import NamedDict, Atts

# todo : remote distribution, visualizing, testing

_local_port = 12345

__all__ = [
    "TrainParameters",
    "TorchTrainer",
]

T_co = TypeVar('T_co', covariant=True)
D = TypeVar('D', Dataset, ndarray, DataFrame)
I = TypeVar('I', int, str)

class TrainParameters(Atts):
    def __init__(
            self,
            epochs:int,
            batch_size:int = None,
            checkpoint:int = 0,
            checkpoint_path:str = "./",
            *args, **kwargs
        ) -> None:
        super().__init__(args, kwargs)
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint = checkpoint if checkpoint > 0 else epochs
        self.checkpoint_path = checkpoint_path

class Slot(object):
    def __init__(
            self,
            name:str,
            model:TorchModel,
            loader:Union[DataLoader, Iterable[DataLoader]],
            device:Union[int, ClientInfo],
            rank:int,
            world_size:int
        ) -> None:
        self.name = name
        self.model = model
        self.loader = loader
        self.device = device
        self.rank = rank
        self.world_size = world_size

    def run(self):
        pass

    def _spawn_process(self, rank:int, size:int, fn:Callable, name, *args, **kwargs) -> None:
        p = mp.Process(target=fn, name=name, args=(rank, size, args), kwargs=kwargs, daemon=True)
        dist.init_process_group(backend="nccl", init_method="tcp://")
        return p

class Trainer(ABC):
    r"""Base Trainer Class in charge of data loading and model training with various features"""
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def infer(self):
        pass

class TorchTrainer(Trainer):
    r"""Trainer supporting pytorch model."""

    class TorchSimpleSet(Dataset):
        def __init__(self, data:Tensor) -> None:
            self.data = data
        
        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, index: int) -> T_co:
            return self.data[index]

    def __init__(
            self,
            model:Union[TorchModel, Tuple[TorchModel, str], Iterable[Union[TorchModel, Tuple[TorchModel, str]]]]=None,
            loader:Union[DataLoader, Tuple[DataLoader, str], Iterable[Union[DataLoader, Tuple[DataLoader, str]]]] = None,
            data:Union[D, Tuple[D, str], Sequence[Union[D, Tuple[D, str]]]]=None,
            device:Union[str, Iterable[int], Iterable[ClientInfo]]=None,
            tp:TrainParameters=None
    ):
        self.slots = NamedDict()
        # Model
        self.hp = NamedDict()
        self.__initial_state = NamedDict()
        self.model = NamedDict()
        if model is not None:
            self.register_model(model)

        # Environment
        self.__host = NamedDict()
        if isinstance(device, str): #Basic (single machine)
            self.__host.insert((ClientInfo('localhost', self._local_port)))
            try:
                self.device = torch.device(device)
            except:
                print(f"Device: {device} not available.")
                self.device = torch.device("cpu")
        else:
            if isinstance(device[0], int): #Single Machine Multi-GPU
                self.__host.insert([ClientInfo('localhost', self._local_port) for i in range(len(device))])
                try:
                    self.device = torch.device('cuda')
                    #If specifying GPU feature integrated to pytorch, this needs to be rewritten
                    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(device)
                except: 
                    print(f"Device: {device} not available.")
            else: #Multi(Remote) Machines
                self.__host.insert(device)
                # Validate Devices Later  

        # Data
        def to_df(data):
                if isinstance(data, ndarray):
                    return self.TorchSimpleSet(torch.tensor(data))
                elif isinstance(data, DataFrame):
                    return self.TorchSimpleSet(torch.tensor(data.values))
                else:
                    return data
        # dataset
        self.dataset = NamedDict()
        self.samplers = NamedDict()
        
        if isinstance(data, D):
            self.dataset.insert(to_df(data))
        elif isinstance(data, tuple):
            self.dataset.insert(to_df(data[0]), [data[1]])
        else:
            if isinstance(data[0], D):
                for d in data:
                    self.dataset.insert(to_df(d))
            else:
                for t in data:
                    self.dataset.insert(to_df(t[0]), [t[1]])
        
        #dataloaders
        self.loaders = NamedDict()
        if isinstance(loader, DataLoader):
            self.loaders.insert(loader)
        elif isinstance(loader, tuple):
            self.loaders.insert(loader[0], [loader[1]])
        else:
            if isinstance(loader[0], DataLoader):
                for l in loader:
                    self.loaders.insert(l)
            else:
                for t in loader:
                    self.loaders.insert(t[0],[t[1]])
        
        self.result = NamedDict()

    def create_slot(self, name:str=None, *args, **kwargs):
        self.slots.insert(Slot(args, kwargs), [name])

    def delete_slot(self, index:I):
        self.slots.drop(I)

    def register_model(
            self,
            model:Union[TorchModel, Tuple[TorchModel, str], Iterable[Union[TorchModel, Tuple[TorchModel, str]]]],
        ):
        if isinstance(model, TorchModel):
            self.model.insert(model)
            self.__initial_state.insert(model.__state__(),[])
            model.trainer = self
        elif isinstance(model, tuple):
            self.model.insert(DDP(model[0]),meta=model[1])
        else: #List
            if isinstance(model[0], TorchModel):
                for m in model:
                    self.model.insert(m)
            elif isinstance(model[0], tuple):
                for t in model:
                    self.model.insert(t[0], t[1])
        return self

    def drop_model(self, index:Union[I, Iterable[I]]) -> None:
        self.model.drop(index)

    def add_data(self, data:Union[D, Iterable[D]], meta:Union[I, Iterable[I]]) -> None:
        if isinstance(data, ndarray):
            self.dataset.insert(self.TorchSimpleSet(torch.tensor(data)), meta)
        elif isinstance(data, DataFrame):
            self.dataset.insert(self.TorchSimpleSet(torch.tensor(data.values)), meta)
        elif isinstance(data, Dataset):
            self.dataset.insert(data, meta)
        else:
            for i in range(len(data)):
                if isinstance(data[i], ndarray):
                    self.dataset.insert(self.TorchSimpleSet(torch.tensor(data[i])), meta[i])
                elif isinstance(data[i], DataFrame):
                    self.dataset.insert(self.TorchSimpleSet(torch.tensor(data[i].values)), meta[i])
                elif isinstance(data[i], Dataset):
                    self.dataset.insert(data[i], meta[i])

    def drop_dataset(self, index:Union[I, Iterable[I]]) -> None:
        self.dataset.drop(index)

    def drop_sampler(self, index:Union[I, Iterable[I]]) -> None:
        self.model.drop(index)

    def drop_loader(self, index:Union[I, Iterable[I]]) -> None:
        self.model.drop(index)

    def link(self, p:Slot, model_index:I=None, loader_index:I=None, parm_index:I=None):
        raise NotImplementedError()
        p.model = self.model

    def all(self, action:str):
        raise NotImplementedError()

    def train(self, index:I, bar:bool=True, loss_graph:bool=False):
        raise NotImplementedError()

    def test(self, index:I, bar:bool=True, loss_graph:bool=False):
        raise NotImplementedError()

    def infer(self, bar:bool=True):
        raise NotImplementedError()

    def train_n_test(self):
        raise NotImplementedError()

    def grid_search(self):
        raise NotImplementedError()

    def kfold(self):
        raise NotImplementedError()