import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from myAdagrad import myAdagrad
from SCAdagrad import SCAdagrad
seed =1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



net= torchvision.models.resnet18()
net = net.to(device)

net1= torchvision.models.resnet18()
net1 = net1.to(device)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adagrad(net.parameters(), lr=0.001)
#optimizer  = optim.Adam(net.parameters(), lr =0.0001)
#optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9)
optimizer = myAdagrad(net.parameters(), lr=0.01)
#optimizer = SCAdagrad(net.parameters(), lr=0.001)


for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    loss_store_myada = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            loss_store_myada.append(running_loss/2000)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training myAdagrad')



##adam training

criterion1 = nn.CrossEntropyLoss()
#optimizer = optim.Adagrad(net.parameters(), lr=0.001)
optimizer1  = optim.Adam(net.parameters(), lr =0.001)
#optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum=0.9)
#optimizer = myAdagrad(net.parameters(), lr=0.01)
#optimizer = SCAdagrad(net.parameters(), lr=0.001)


for epoch in range(200):  # loop over the dataset multiple times

    running_loss1 = 0.0
    loss_store_adam = []
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion1(outputs, labels)
        loss.backward()
        optimizer1.step()

        # print statistics
        running_loss1 += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            loss_store_adam.append(running_loss/2000)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training adam')