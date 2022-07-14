import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
from torchvision import datasets, transforms
import model
from trainer import TorchTrainer
from interface import load_config
import ssl

#CIFAR download error without this
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 500
config = load_config('example.yml', 'yml')
hp = model.Atts(config.hp)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_n = torch.cuda.device_count()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader =  DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, padding_mode='reflect', bias=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2)
        )
        self.lf = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 100, bias=True)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = torch.flatten(x, 1)
        x = self.lf(x)
        return x

class Eval(model.Metric):
    def eval(self, pred: torch.Tensor, target: torch.Tensor):
        print(pred, target)

net = NN()
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(),lr=hp.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch:hp.ld**epoch)
model = model.TorchModel(net, hp=hp, eval=Eval(), cr=criterion, opt=opt, sch=scheduler)
agent = TorchTrainer(model, trainloader=trainloader, testloader=testloader)
#agent.train()
agent.infer()