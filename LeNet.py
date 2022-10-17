import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchsummary
import json

class Dataset:
    def __init__(self, config):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if config["LOADTRAIN"]:
            self.trainset = torchvision.datasets.CIFAR10(root=config["DATAROOT"], train=True,download=False, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=config["BATCHSIZE"],shuffle=True, num_workers=2)
        if config["LOADTEST"]:
            self.testset = torchvision.datasets.CIFAR10(root=config["DATAROOT"], train=False,download=False, transform=transform)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=config["BATCHSIZE"],shuffle=False, num_workers=2)

class CNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(3, 6, 5) # input 4x3x32x32 (batch=4, 3 ch), with 6 kernels (3 ch), kernel size 5x5, output 4x6x28x28(6 ch, batch=4, 32-5+1=28)
        self.pool = nn.MaxPool2d(2, 2) # output 4x6x14x14
        self.conv2 = nn.Conv2d(6, 16, 5) # output 4x16x10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 4x6x14x14
        x = self.pool(F.relu(self.conv2(x))) # 4x16x5x5
        x = torch.flatten(x, 1) # 4x400 flatten all dimensions except batch 
        x = F.relu(self.fc1(x)) # 4x120
        x = F.relu(self.fc2(x)) # 4x84
        x = self.fc3(x) # 4x10
        return x

    def summarize(self, config, input=(3,32,32)):
        torchsummary.summary(self,input,batch_size=config["BATCHSIZE"])

    def train(self, config):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=config["LR"], momentum=config["MOMENTUM"])
        num_batch = len(self.dataset.trainloader)
        epoch_loss = []
        for epoch in range(config["EPOCHS"]): 
            running_loss = 0.0
            for i, data in enumerate(self.dataset.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print("current epoch: %2d, mean loss: %.3f"%(epoch+1,running_loss/num_batch))
            epoch_loss.append(running_loss/num_batch)

        print('------------Finished Training------------')
        torch.save(self.state_dict(), "./model/LeNet_model_%d.pth"%config["TRAINID"])
        with open("./model/LeNet_config_%d.json"%config["TRAINID"],"w") as outfile:
            json.dump(config, outfile, indent = 4)
        np.savez('./result/LeNet_model_%d.npz'%config["TRAINID"], data=np.array(epoch_loss))

    def test(self, config):
        self.load_state_dict(torch.load("./model/LeNet_model_%d.pth"%config["TRAINID"]))
        correct = total = 0
        with torch.no_grad():
            for data in self.dataset.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.forward(images)
                # the class with the highest output is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the 10000 test images:",'{:.2%}'.format(correct/total))
        return correct/total


if __name__=="__main__":

    L=[]

    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":1,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":256,
        "LR":0.003,
        "EPOCHS":100,
        "MOMENTUM":0.9
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))

    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":2,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":128,
        "LR":0.001,
        "EPOCHS":100,
        "MOMENTUM":0.9
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))

    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":3,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":512,
        "LR":0.005,
        "EPOCHS":100,
        "MOMENTUM":0.8
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))

    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":4,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":256,
        "LR":0.001,
        "EPOCHS":100,
        "MOMENTUM":0.8
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))

    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":5,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":64,
        "LR":0.001,
        "EPOCHS":100,
        "MOMENTUM":0.9
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))


    config = \
    {
        "DATAROOT":"/home/featurize/data", 
        "TRAINID":6,
        "LOADTRAIN":True,
        "LOADTEST":True,
        "BATCHSIZE":32,
        "LR":0.003,
        "EPOCHS":100,
        "MOMENTUM":0.9
    }

    dataset = Dataset(config)
    net = CNN(dataset)
    # net.summarize(config)
    L.append(net.test(config))

    np.savez('./result/LeNet_test_res.npz', data=np.array(L))





