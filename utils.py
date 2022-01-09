import matplotlib.pyplot as plt
import torchvision as tv
import torchvision
from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
import os

def get_loaders_MNIST(batch_size=64):
    transforms =tv.transforms.Compose([tv.transforms.ToTensor(),
                                    tv.transforms.Normalize((0.1307,), (0.3081,))])
    train_data = tv.datasets.MNIST( root="./data/",
                                train=True,
                                download=True,
                                transform = transforms)

    test_data = tv.datasets.MNIST( root="./data/",
                                train=False,
                                download=True,
                                transform = transforms)

    train_loader = DataLoader(train_data,
                    batch_size=batch_size, shuffle=True,
                    drop_last=True,num_workers=2)

    test_loader = DataLoader(dataset = test_data, 
                                batch_size = batch_size,
                                shuffle = False)
    return train_loader, test_loader

def get_loaders_CIFAR10(batch_size=64):
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform= transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return train_loader, test_loader

def decet(feature,targets,save_path,epoch):
    color = ["red", "black", "yellow", "green", "pink",
    "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()
    for j in cls:
        mask = [targets == j]
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right") 
        plt.title("epoch={}".format(str(epoch+1)))
    plt.savefig(save_path)
    plt.draw()
    plt.pause(0.01)

def plot3D(embeds, labels,fig_path,epoch):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")
    plt.title(f'epoch={epoch}')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.draw()
    plt.pause(0.01)


def setupImageFolders(CIFAR10,Latent_dim):
    import os, shutil
    folder = f'./Images/{"CIFAR10" if CIFAR10 else "MNIST"}-{Latent_dim}D'
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def testdata_accuracy(device,arcnet,test_loader,latent_dim,epoch,img_path):

    acc = 0
    embeddings = torch.zeros(1,latent_dim).to(device)
    targets = torch.zeros(1).to(device)
    Ws = torch.zeros(1,latent_dim,2)
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        predictions,__ = arcnet(x,y) 
        #---------test accuracy-----------------
        value = torch.argmax(predictions, dim=1)
        acc += torch.sum((value == y).float())

        embedding,weights = arcnet.getEmbbedings(x)
        embeddings = torch.cat([embeddings,embedding])
        targets = torch.cat([targets,y])

    if latent_dim == 2:
        decet(embeddings[1:].data.cpu(),targets[1:].data.cpu(),img_path,epoch)
    else:
        plot3D(embeddings[1:].data.cpu(),targets[1:].data.cpu(),img_path,epoch)

    return acc.item() / (len(targets)-1)

