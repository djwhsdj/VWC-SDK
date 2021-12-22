import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
from utils import *
import logging
import pandas as pd
from pandas import Series, DataFrame

GPU_NUM = 3 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        MNIST('/Data', train=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]), download=True), batch_size=128, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('/Data', train=False, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ]),download=True), batch_size=128, num_workers=0)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = Conv2d_Q_(1, 1, 6, kernel_size=5, padding=0, stride=1 , bias=False)
        self.conv2 = Conv2d_Q_(1, 6, 16, kernel_size=5, padding=0, stride=1 , bias=False)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.act = Activate(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.act(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

learning_rate = 0.001
epochs = 20

def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        cnt = 0
        loss_sum = 0
        loss_ = 0
        for i, (img, target) in enumerate(train_loader):
            cnt += 1
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            # if i%10==0 and 500 :
            #     print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, epochs, i, len(train_loader), loss.item()))
            loss_sum += loss.item()
        
        loss_sum = loss_sum / cnt
        loss_= round(loss_sum, 4)
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch+1, epochs, acc))
        

        if best_acc<acc:
            torch.save( model, './mnist.pth')
            print('Model saved!!')
            best_acc=acc
    print("Best Acc=%.4f"%(best_acc))

def main():
    train_loader, test_loader = get_dataloader()
    model = Net().cuda()
    print(model)
    train_model(model, train_loader, test_loader)

if __name__=='__main__':
    main()
