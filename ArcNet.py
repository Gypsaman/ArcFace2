import torch.nn.functional as F
import torchvision as tv
import torch.nn as nn
import torch
from Network import Embedding,ConvNet

class ArcNet(nn.Module):
    def __init__(self,num_classes,latent_dim,CIFAR10=False,s=20,m=0.1): # orig, s=10,m=0.1
        super().__init__()
        self.embedding = Embedding(latent_dim,CIFAR10)
        self.s = s  # scale
        self.m = torch.tensor(m)  # margin
        
        self.fc = torch.nn.Linear(latent_dim,num_classes)
        self.PredActivation = torch.nn.Softmax(dim=1)
        
        self.eps = 1.0e-7

    def forward(self, x, labels):
        embedding = self.embedding(x)

        self.fc.weight = torch.nn.parameter.Parameter(torch.nn.functional.normalize(self.fc.weight))
        wf = self.fc(embedding)
        
        prediction = self.PredActivation(wf)

        wf_m = torch.cos(
            torch.acos(
                torch.clamp(wf,
                    min=-1+self.eps,
                    max=1-self.eps
                )
             )+self.m)
        
        numerator = self.s * (torch.diagonal(wf_m.transpose(0, 1)[labels]))
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
        return prediction, loss
    
    
    def getEmbbedings(self,x):
        with torch.no_grad():
            embedding = F.normalize(self.embedding(x),dim=1)
            weights = F.normalize(self.fc.weight,dim=1)
        return embedding,weights

class ArcNetAhmed(nn.Module):
    def __init__(self,num_classes,latent_dim,CIFAR10=False,s=20,m=0.1): # orig, s=10,m=0.1
        super().__init__()
        self.embedding = Embedding(latent_dim,CIFAR10)
        self.s = s  # scale
        self.m = torch.tensor(m)  # margin
        # self.w=torch.rand(latent_dim,num_classes)
        self.fc = torch.nn.Linear(latent_dim,num_classes)
        self.PredActivation = torch.nn.Softmax(dim=1)
        # self.loss = torch.nn.CrossEntropyLoss()
        self.eps = 1.0e-7
        self.arcloss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        embedding = self.embedding(x)
        embedding = F.normalize(embedding,dim=1) # normalize latent output

        self.fc.weight = torch.nn.parameter.Parameter(torch.nn.functional.normalize(self.fc.weight))
        wf = self.fc(embedding)

        prediction = self.PredActivation(wf)

        cos_theta_m= torch.cos(
            torch.acos(
                torch.clamp(wf,
                    min=-1+self.eps,
                    max=1-self.eps
                )
             )+self.m)

        excl = torch.cat([torch.cat([wf[i, :z],cos_theta_m[i,z].reshape(-1), wf[i, z+1:]]).unsqueeze(0) for i, z in enumerate(labels)], dim=0)

        excl = excl * self.s
        loss = self.arcloss(excl,labels)

        return prediction, loss
    
    
    def getEmbbedings(self,x):
        with torch.no_grad():
            embedding = F.normalize(self.embedding(x),dim=1)
            weights = F.normalize(self.fc.weight,dim=1)
        return embedding,weights



