import utils
from ArcNet import ArcNetAhmed
import torch

CIFAR10 = False
save_pic_path = "./Images"
device = torch.device('cpu')
latent_dim = 2
num_classes = 10

train_loader,test_loader = utils.get_loaders_CIFAR10() if CIFAR10 else utils.get_loaders_MNIST()

arcnet = ArcNetAhmed(num_classes, latent_dim,CIFAR10).to(device)

arcnet.load_state_dict(torch.load('./trainedmodel/model-MNIST-2D.pth'))
arcnet.eval()

img_path = f'./Images/test/w-test.jpg'
utils.testdata_accuracy(device,arcnet,test_loader,latent_dim,9999,img_path=img_path)