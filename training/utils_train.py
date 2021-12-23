import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

from optimizers import fgsmm
import numpy as np

def get_data_loaders(batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
    datasets.KMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.KMNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=test_batch_size, shuffle=True)
    return train_loader, test_loader

def get_data_loaders_cifar(batch_size, test_batch_size):
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
    return trainloader, testloader

def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
            
def train_at(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data.requires_grad_(True)
        v = torch.zeros_like(data)
        xv = (data, v)

        def adv_loss(x, y = target):
            return -F.nll_loss(model(x), y)

        xx, mmsgf = fgsmm(adv_loss, xv, T = 1, lr = 0.075, gamma = 0.)
        
        output = model(torch.cat((data,xx[0]), 0))
        
        loss = F.nll_loss(output, torch.cat((target, target), 0))
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
def train_alp(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data.requires_grad_(True)
        v = torch.zeros_like(data)
        xv = (data, v)

        def adv_loss(x, y = target):
            return -F.nll_loss(model(x), y)

        xx, mmsgf = fgsmm(adv_loss, xv, T = 1, lr = 0.075, gamma = 0.)
        
        output = model(data)
        
        loss = F.nll_loss(output, target) + torch.mean(torch.abs(model.logits(data) - model.logits(xx[0])))
        
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
def train_atpx(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    
    def energy(x):
        return -torch.logsumexp(model.logits(x), dim=1)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data.requires_grad_(True)
        v = torch.zeros_like(data)
        xv = (data, v)

        def adv_loss(x, y = target):
            return -F.nll_loss(model(x), y)

        xx, mmsgf = fgsmm(adv_loss, xv, T = 1, lr = 0.075, gamma = 0.)
        
        output = model(torch.cat((data,xx[0]), 0))
        
        loss = F.nll_loss(output, torch.cat((target, target), 0)) + torch.abs(energy(xx[0]).mean() - energy(data).mean())
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
def train_ara(model, device, train_loader, optimizer, epoch, train_losses):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data.requires_grad_(True)
        v = torch.zeros_like(data)
        xv = (data, v)

        def adv_loss(x, y = target):
            return -F.nll_loss(model(x), y)
        
        xxs = []
        N = 1
        # To do N > 1, we would have to use a Bayesian model to avoid overfitting.
        for _ in range(N):
            T = 1 + np.random.poisson(1)
            #T = 5
            lr_min, lr_max = 0.05, 0.15
            lr_a = np.random.beta(1, 1)*(lr_max - lr_min) + lr_min
            #lr = 0.05
            gamma = 0.0

            xx, mmsgf = fgsmm(adv_loss, xv, T = T, lr = lr_a, gamma = gamma)
            xxs.append(xx[0])
            
        xx = torch.cat(tuple(xxs), 0)

        output = model(torch.cat((data,xx), 0))
        
        loss = F.nll_loss(output, torch.cat((target, target.repeat(N)), 0)) 
        
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
def train_arapx(model, device, train_loader, optimizer, epoch, train_losses):
    
    def energy(x):
        return -torch.logsumexp(model.logits(x), dim=1)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data.requires_grad_(True)
        v = torch.zeros_like(data)
        xv = (data, v)

        def adv_loss(x, y = target):
            return -F.nll_loss(model(x), y)
        
        xxs = []
        N = 1
        # To do N > 1, we would have to use a Bayesian model to avoid overfitting.
        for _ in range(N):
            T = 1 + np.random.poisson(1)
            #T = 5
            lr_min, lr_max = 0.05, 0.15
            lr_a = np.random.beta(1, 1)*(lr_max - lr_min) + lr_min
            #lr = 0.05
            gamma = 0.0

            xx, mmsgf = fgsmm(adv_loss, xv, T = T, lr = lr_a, gamma = gamma)
            xxs.append(xx[0])
            
        xx = torch.cat(tuple(xxs), 0)

        output = model(torch.cat((data,xx), 0))
        
        loss = F.nll_loss(output, torch.cat((target, target.repeat(N)), 0)) + torch.abs(energy(xx).mean() - energy(data).mean())
        
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))