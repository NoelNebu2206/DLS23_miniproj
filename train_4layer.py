import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchviz
import torchsummary
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchviz import make_dot

from torchvision.models.resnet import ResNet18_Weights

import matplotlib.pyplot as plt

import os
import argparse
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epochprint('==> Preparing data..')

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

base_model = resnet18(num_classes=10)

x = iter(trainloader)
images, label = x.__next__()
y = base_model(images)

from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=1000, num_hidden=64):
        super(ModifiedResNet18, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, num_hidden, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.bn1 = nn.BatchNorm2d(num_hidden)

        self.model.layer1[0].conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn1 = nn.BatchNorm2d(num_hidden)
        self.model.layer1[0].conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[0].bn2 = nn.BatchNorm2d(num_hidden)

        self.model.layer1[1].conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn1 = nn.BatchNorm2d(num_hidden)
        self.model.layer1[1].conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer1[1].bn2 = nn.BatchNorm2d(num_hidden)

        self.model.layer2[0].conv1 = nn.Conv2d(num_hidden, num_hidden*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer2[0].bn1 = nn.BatchNorm2d(num_hidden*2)
        self.model.layer2[0].conv2 = nn.Conv2d(num_hidden*2, num_hidden*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[0].bn2 = nn.BatchNorm2d(num_hidden*2)
        self.model.layer2[0].downsample[0] = nn.Conv2d(num_hidden, num_hidden*2, kernel_size=2, stride=2, padding=0, bias=False)
        self.model.layer2[0].downsample[1] = nn.BatchNorm2d(num_hidden*2)
        

        self.model.layer2[1].conv1 = nn.Conv2d(num_hidden*2, num_hidden*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn1 = nn.BatchNorm2d(num_hidden*2)
        self.model.layer2[1].conv2 = nn.Conv2d(num_hidden*2, num_hidden*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer2[1].bn2 = nn.BatchNorm2d(num_hidden*2)

        self.model.layer3[0].conv1 = nn.Conv2d(num_hidden*2, num_hidden*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer3[0].bn1 = nn.BatchNorm2d(num_hidden*4)
        self.model.layer3[0].conv2 = nn.Conv2d(num_hidden*4, num_hidden*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[0].bn2 = nn.BatchNorm2d(num_hidden*4)
        self.model.layer3[0].downsample[0] = nn.Conv2d(num_hidden*2, num_hidden*4, kernel_size=2, stride=2, padding=0, bias=False)
        self.model.layer3[0].downsample[1] = nn.BatchNorm2d(num_hidden*4)

        self.model.layer3[1].conv1 = nn.Conv2d(num_hidden*4, num_hidden*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn1 = nn.BatchNorm2d(num_hidden*4)
        self.model.layer3[1].conv2 = nn.Conv2d(num_hidden*4, num_hidden*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer3[1].bn2 = nn.BatchNorm2d(num_hidden*4)

        self.model.layer4[0].conv1 = nn.Conv2d(num_hidden*4, num_hidden*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.layer4[0].bn1 = nn.BatchNorm2d(num_hidden*8)
        self.model.layer4[0].conv2 = nn.Conv2d(num_hidden*8, num_hidden*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[0].bn2 = nn.BatchNorm2d(num_hidden*8)
        self.model.layer4[0].downsample[0] = nn.Conv2d(num_hidden*4, num_hidden*8, kernel_size=2, stride=2, padding=0, bias=False)
        self.model.layer4[0].downsample[1] = nn.BatchNorm2d(num_hidden*8)

        self.model.layer4[1].conv1 = nn.Conv2d(num_hidden*8, num_hidden*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn1 = nn.BatchNorm2d(num_hidden*8)
        self.model.layer4[1].conv2 = nn.Conv2d(num_hidden*8, num_hidden*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.layer4[1].bn2 = nn.BatchNorm2d(num_hidden*8)
        
        
        self.model.fc = nn.Linear(in_features=num_hidden*8, out_features=num_classes)
        

    def forward(self, x):
        x = self.model(x)
        return x


my_model = ModifiedResNet18(10, 39)
num_trainable_params_new = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
print(num_trainable_params_new)

print("Summary second time", torchsummary.summary(my_model, input_size =(3,32,32), device='cpu'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# Define your model
model = ModifiedResNet18(10, 39)
model = nn.DataParallel(model).cuda()

test_acc = []
training_loss = []

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr_max = 0.045
wd = 0.005

optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

num_epochs = 55 
lr_schedule = lambda t: np.interp([t], [0, num_epochs*2//5, num_epochs*4//5, num_epochs], [0, lr_max, lr_max/20.0, 0])[0]
scaler = torch.cuda.amp.GradScaler()

# Train your model
for epoch in range(num_epochs):
    running_loss = 0.0
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        model.train()

        # Get the inputs and labels from the data loader
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        lr = lr_schedule(epoch + (i + 1)/len(trainloader))
        optimizer.param_groups[0].update(lr=lr)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimization
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Print statistics
        running_loss += loss.item()
        train_loss += loss.item()
        if i % 100 == 99:
            print('[EPOCH: %d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    training_loss.append(train_loss/100)
    
    # Test your model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            # Get the inputs and labels from the data loader
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
    
            # Forward pass and prediction
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
    
            # Compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    test_acc.append(100*correct/total)

# Plot test accuracy vs epoch
plt.plot(range(1,num_epochs+1), test_acc)
plt.title('Test Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.show()

# Plot training loss vs epoch
plt.plot(range(1,num_epochs), training_loss)
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()
