import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# batch_size = 4
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)


if __name__=="__main__":
    # image,label=list(trainloader)[0] # 4x3x32x32
    # print(image.shape)
    # print(label)
    # print(len(list(trainloader)))
    a=np.array([[1, 2, 3], [4, 5, 6]])
    b=np.array([1, 2])
    np.savez('./123.npz', a=a, b=b)
    data = np.load('./123.npz')
    print(data['a'])




    # conv1=nn.Conv2d(3, 6, 5)
    # conv2 = nn.Conv2d(6, 16, 5)
    # pool=nn.MaxPool2d(2, 2)

    # image = pool(F.relu(conv1(image))) # 4x6x14x14
    # image = pool(F.relu(conv2(image))) # 4x16x5x5
    # image = torch.flatten(image, 1) # 4x400
    # print(image.shape)

    




