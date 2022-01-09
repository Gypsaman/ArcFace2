import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, latent_dim,CIFAR10=False):
        super().__init__()
        self.inputlayer = nn.Conv2d(3, 64, 3, 2, 1) if CIFAR10 else nn.Conv2d(1, 64, 3, 2, 1)
        self.cnn_layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,3, 2, 1),
            nn.ReLU())
        self.linear_layer = nn.Linear(16*4*4,latent_dim)
        #self.output_layer = nn.Linear(latent_dim,10,bias=False)

    def forward(self, xs):
        cnn_initial = self.inputlayer(xs)
        cnn_out = self.cnn_layers(cnn_initial)
        flatten = cnn_out.reshape(-1,16*4*4)
        latent_out = self.linear_layer(flatten)
        #output = self.output_layer(latent_out)
        return latent_out

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc_projection = nn.Linear(512, 3)

