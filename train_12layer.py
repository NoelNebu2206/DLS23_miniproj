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

print(torchsummary.summary(base_model, input_size =(3,32,32), device='cpu'))

from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn

class ModifiedResNet18_with_true_12_layers(nn.Module):
    def __init__(self, num_classes=1000, num_hidden=64, layers=[2,1,2,1,2,1,2,1,2,1,1,1]):
        super(ModifiedResNet18_with_true_12_layers, self).__init__()
        self.in_channels = num_hidden

        self.conv1 = nn.Conv2d(3, num_hidden, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block=BasicBlock, out_channels=num_hidden*2, blocks=layers[0],stride=2)
        self.layer2 = self._make_layer(block=BasicBlock, out_channels=num_hidden*2, blocks=layers[1], stride=1)
        self.layer3 = self._make_layer(block=BasicBlock, out_channels=num_hidden*4, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=BasicBlock, out_channels=num_hidden*4, blocks=layers[3], stride=1)
        self.layer5 = self._make_layer(block=BasicBlock, out_channels=num_hidden*6, blocks=layers[4], stride=1)
        self.layer6 = self._make_layer(block=BasicBlock, out_channels=num_hidden*6, blocks=layers[5], stride=2)
        self.layer7 = self._make_layer(block=BasicBlock, out_channels=num_hidden*7, blocks=layers[6], stride=1)
        self.layer8 = self._make_layer(block=BasicBlock, out_channels=num_hidden*7, blocks=layers[7], stride=1)
        self.layer9 = self._make_layer(block=BasicBlock, out_channels=num_hidden*8, blocks=layers[8], stride=1)
        self.layer10 = self._make_layer(block=BasicBlock, out_channels=num_hidden*8, blocks=layers[9], stride=2)
        self.layer11 = self._make_layer(block=BasicBlock, out_channels=num_hidden*9, blocks=layers[10], stride=1)
        self.layer12 = self._make_layer(block=BasicBlock, out_channels=num_hidden*10, blocks=layers[11], stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_hidden*10 * BasicBlock.expansion, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

my_model = ModifiedResNet18_with_true_12_layers(10, 20)
num_trainable_params_new = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
print("New trainable parameters are",num_trainable_params_new)

print("Summary second time", torchsummary.summary(my_model, input_size =(3,32,32), device='cpu'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# Define your model
model = ModifiedResNet18_with_true_12_layers(10, 20)
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
