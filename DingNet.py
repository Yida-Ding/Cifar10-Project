import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchsummary
import json
from BatchNorm import BatchNorm

class Dataset:
    def __init__(self, config):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if config["LOADTRAIN"]:
            self.trainset = torchvision.datasets.CIFAR10(root=config["DATAROOT"], train=True,download=False, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=config["BATCHSIZE"],shuffle=True, num_workers=2)  # type: ignore
        if config["LOADTEST"]:
            self.testset = torchvision.datasets.CIFAR10(root=config["DATAROOT"], train=False,download=False, transform=transform)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=config["BATCHSIZE"],shuffle=False, num_workers=2)  # type: ignore

class CNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.bn1 = BatchNorm(256, num_dims=2)
        self.bn2 = BatchNorm(128, num_dims=2)
        self.bnc1 = nn.BatchNorm2d(32)
        self.bnc2 = nn.BatchNorm2d(64)
        self.bnc3 = nn.BatchNorm2d(128)
        self.bnc4 = nn.BatchNorm2d(256)
        self.bnc5 = nn.BatchNorm2d(512)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.bnc2(self.conv2(F.relu(self.bnc1(self.conv1(x)))))))
        x = self.pool(F.relu(self.bnc4(self.conv4(F.relu(self.bnc3(self.conv3(x)))))))
        x = self.pool(F.relu(self.bnc5(self.conv5(x))))
        x = torch.flatten(x, 1) 
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
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
        torch.save(self.state_dict(), "./model/DingNet_model_%d.pth"%config["TRAINID"])
        with open("./model/DingNet_config_%d.json"%config["TRAINID"],"w") as outfile:
            json.dump(config, outfile, indent = 4)
        np.savez('./result/DingNet_model_%d.npz'%config["TRAINID"], data=np.array(epoch_loss))

    def test(self):
        # self.load_state_dict(torch.load("./model/DingNet_model_%d.pth"%config["TRAINID"]))
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
    
    test_res = []

    for i in range(1,7):

        with open("./model/DingNet_config_%d.json"%i,'r') as outfile:
            config = json.load(outfile)
            dataset = Dataset(config)
            net = CNN(dataset)
            net.summarize(config)
            net.train(config)
            test_res.append(net.test())
    
    np.savez('./result/DingNet_test_res_bnlc.npz', data=np.array(test_res))



