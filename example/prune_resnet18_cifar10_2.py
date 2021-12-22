import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import ResNet18, ResNet18_Q
import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
from utils import *
import logging
import pandas as pd
from pandas import Series, DataFrame

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

GPU_NUM = 2 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=400) # 40 # 200
parser.add_argument('--step_size', type=int, default=125) # 65
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001) # prune 0.00001 # 원래 0.001
parser.add_argument('--q', type=int, default=0) # 0 : full, 1 : quantize
parser.add_argument('--ab', type=int, default=0)
parser.add_argument('--wb', type=int, default=0)
parser.add_argument('--optim', type=str, default = 'Adam')
parser.add_argument('--directory', type=str, default = '/Data') # fc layer 크기

args = parser.parse_args()
if args.q == 0:
    file_name = './log/512x128_fc_resnet18_original_schedule_lr%.3f_e%d.log'%(args.lr, args.total_epochs)
else :
    file_name = './log/512_128_fc_resnet18_Qs_schedule_ab_%d_wb_%d_lr%.3f_e%d.log'%(args.ab, args.wb, args.lr, args.total_epochs)
file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

prune_root = '/Data/Checkpoint1'

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('/Data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]), download=True),batch_size=args.batch_size, num_workers=1) 
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('/Data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),download=True),batch_size=args.batch_size, num_workers=1) 
    return train_loader, test_loader

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

def train_model(model, train_loader, test_loader):
    
    if args.optim == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else :    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    result_list = {'train_loss' : [], 'train_acc' : []}
    for epoch in range(args.total_epochs):
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
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
            loss_sum += loss.item()

        loss_sum  = loss_sum / cnt
        loss_ = round(loss_sum, 4)    
        model.eval()
        acc = eval(model, test_loader)
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))

        result_list['train_loss'].append(loss_)
        result_list['train_acc'].append(acc)

        data_frame = pd.DataFrame(data=result_list, index=range(0, epoch+1))
        if args.q == 0 :
            csv_name = args.directory + "/csv1/512x128_fc_resnet18_original_schedule_lr%.3f_e%d.csv"%(args.lr , args.total_epochs) # Quan 추가. 나중에 수정해야함
        else :
            csv_name = args.directory + '/csv1/512x128_fc_resnet18_Qs_schedule_ab_%d_wb_%d_lr%.3f_e%d.csv'%(args.ab, args.wb, args.lr, args.total_epochs)
        data_frame.to_csv(csv_name, index_label='epoch')        

        if best_acc<acc:
            if args.q == 0 :
                torch.save( model, args.directory +'/Checkpoint1/512x128_fc_resnet18_schedule_original_lr%.3f_e%d_round%d.pth'%(args.lr, args.total_epochs, args.round) )
            else :
                torch.save( model, args.directory +'/Checkpoint1/512x128_fc_resnet18_Qs_schedule_ab_%d_wb_%d_lr%.3f_e%d_round%d.pth'%(args.ab, args.wb, args.lr, args.total_epochs, args.round) )
            print('Model saved!!')
            best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))

def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    def prune_conv(conv, to_p_amount=0):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, to_p_amount=to_p_amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    images = [16, 16, 16, 16, 8, 8, 4, 4]
    IC = [64, 64, 128, 128, 256, 256, 512, 512]
    array = [512, 128]
    block_pruned = []
    block_pruned1 = []
    tiled_ic = []
    after_ic = []
    
    for i in range(len(images)) :
        t_IC = vw_sdk(images[i], images[i], 3, 3, IC[i], IC[i], array[0], array[1])
        res_block = IC[i] % t_IC
        after_ic.append(int(IC[i]) - res_block)
        block_pruned.append(res_block)
        block_pruned1.append(res_block)
        tiled_ic.append(t_IC)

    for i in range(len(block_pruned)-1) :
        if after_ic[i] > after_ic[i+1] :
            after_ic[i] = after_ic[i] - tiled_ic[i]
            block_pruned[i] = block_pruned[i] + tiled_ic[i]

    for i in range(len(block_pruned1)) :
        if i%2 != 0 :
            block_pruned1[i] = 0

    print("block_pruned")
    print(block_pruned)
    print("block_pruned1")
    print(block_pruned1)
    print("tiled_ic")
    print(tiled_ic)
    print("after")
    print(after_ic)

    block_pruned.append(0) # for last layer
    block_pruned1.append(0)
    blk_id = 0
    for m in model.modules():
        if args.q != 1 :
            if isinstance( m, resnet.BasicBlock ):
                print(m)
                prune_conv( m.conv1, block_pruned[blk_id] )
                # print(block_pruned[blk_id])
                prune_conv( m.conv2, block_pruned1[blk_id] )
                blk_id+=1
        else :
            if isinstance( m, resnet.BasicBlock_Q ):
                prune_conv( m.conv1, block_pruned[blk_id] )
                prune_conv( m.conv2, block_pruned1[blk_id] )
                blk_id+=1
    print("*"*100)
    return model    

def main():
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        if args.q == 0 :
            model = ResNet18(num_classes=10)
        else :
            model = ResNet18_Q(args.ab, args.wb, num_classes=10)
        
        print(model)
        train_model(model, train_loader, test_loader)
        
    elif args.mode=='prune':
        if args.q == 0 :
            previous_ckpt = prune_root + '/resnet18_schedule_original_lr0.001_e160_round%d.pth'%(args.round-1) 
        else : 
            previous_ckpt = prune_root + '/resnet18_Qs_schedule_ab_%d_wb_%d_lr0.001_e160_round%d.pth'%(args.ab, args.wb, args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        train_model(model, train_loader, test_loader)

    elif args.mode=='test':
        if args.q == 0 :
            ckpt = prune_root + '/resnet18_original_lr%.3f_e%d_round%d.pth'%(args.lr, args.total_epochs, args.round)  # args.round
        else :
            ckpt = prune_root + '/resnet18_Q_ab_%d_wb_%d_lr%.3f_e%d_round%d.pth'%(args.ab, args.wb, args.lr, args.total_epochs, args.round) # args.round
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
