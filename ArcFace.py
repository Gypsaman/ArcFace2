import sys
# from Network import Network
from ArcNet import ArcNet, ArcNetAhmed
import utils
import torch
import torch.nn as nn
import torch as t


  
def SetLearningRate(epoch):

    # rates dictionary {epcoh:learningrate}
    rates = {0:0.1,
            10:0.005,
            20:0.002,
            40:0.001,
            60:0.0001}

    return [v for k,v in rates.items() if k <= epoch][-1]



def process():

    train_loader,test_loader = utils.get_loaders_CIFAR10() if CIFAR10 else utils.get_loaders_MNIST()
    
    arcnet = ArcNetAhmed(num_classes, latent_dim,CIFAR10).to(device)

    # optimizerarc = t.optim.SGD([{'params': arcnet.parameters()}], lr=0.1, momentum=0.9, weight_decay=0.00005)
    optimizerarc = torch.optim.Adam(arcnet.parameters(), lr=0.01)

    num_epochs = 80

    for epoch in range(num_epochs):
        iter_loss = 0.0

        arcnet.train()  

        for iteration, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            #-------------loss calculation--------------
            prediction, arcloss = arcnet(x,y)
            iter_loss += arcloss

            #-------------compute accuracy-------------
            value = t.argmax(prediction, dim=1)
            acc = t.sum((value == y).float()) / len(y)
        
            #----------compute gradients, update network parameters----
            optimizerarc.zero_grad()
            arcloss.backward()
            optimizerarc.step()
            

        # optimizerarc.param_groups[0]['lr'] = SetLearningRate(epoch)

        img_path = f'./Images/{"CIFAR10" if CIFAR10 else "MNIST"}-{latent_dim}D/{epoch+1}.jpg'
        test_acc = utils.testdata_accuracy(device,arcnet, test_loader, latent_dim, epoch,img_path)

        print (f'Epoch {epoch+1}/{num_epochs}, Training Loss: {iter_loss/(iteration+1):.3f}, Training Accuracy: {acc:.3f}, Test Accurary {test_acc:.3f}')

        if test_acc > 0.98 and acc > 0.99:
            break
   
        
     
    PATH = f'./trainedmodel/model-{"CIFAR10" if CIFAR10 else "MNIST"}-{latent_dim}D.pth'

    torch.save(arcnet.state_dict(),PATH)
 
if __name__ == "__main__":
    from time import ctime

    CIFAR10 = False
    save_pic_path = "./Images"
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    latent_dim = 2  # embedding size
    num_classes = 10

    processing_options = [
        (False,2),
        (False,3),
        (True,2),
        (True,3)
        ]

    for CIFAR10, latent_dim in processing_options                   :
        utils.setupImageFolders(CIFAR10,latent_dim)

        print(f'{"CIFAR10" if CIFAR10 else "MNIST"} With {latent_dim} Latent Dimmensions, Using: {device}')

        starttime = ctime()
        print(starttime)

        process()

        print(f'From: {starttime} To: {ctime()}')

    sys.exit(0)
