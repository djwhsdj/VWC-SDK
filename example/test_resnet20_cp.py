import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import resnet20q, resnet20
import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
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


GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=512) 
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=1)
parser.add_argument('--step_size', type=int, default=125) 
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001) # 0.01 best
parser.add_argument('--q', type=int, default=0) # 0 : fsull, 1 : quantize
parser.add_argument('--ab', type=int, default=0)
parser.add_argument('--wb', type=int, default=0)
parser.add_argument('--optim', type=str, default = 'Adam')
parser.add_argument('--directory', type=str, default = '/Data') # fc layer 크기
parser.add_argument('--scheduler', type=str, default = 'Multi_StepLR')
parser.add_argument('--f', type=int, default = 5)

args = parser.parse_args()

if args.q == 0:
    file_name = './log_cifar10/resnet20_e_%d.log'%(args.total_epochs)
else :
    file_name = './log_cifar10/aa.log'
    # file_name = './log_cifar10/512x256_resnet20_ab_%d_wb_%d_e_%d.log'%(args.ab, args.wb, args.total_epochs)
    # file_name = './log_cifar10/512x256_2bit_resnet20_ab_%d_wb_%d_e_%d.log'%(args.ab, args.wb, args.total_epochs)
file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

prune_root = '/Data/ckpt_cifar10'

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('/Data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomVerticalFlip(),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]), download=True), shuffle=True, batch_size=args.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('/Data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]),download=True), shuffle=False, batch_size=args.batch_size, num_workers=0)
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

acc_list = []
def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.optim == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else :    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.scheduler == 'StepLR' :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1) 
    elif args.scheduler == 'Multi_StepLR' :
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 125, 175], gamma=0.1) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1) 
        # 스케쥴러 값 수정해볼 것.
    else :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    model.to(device)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    best_acc = -1
    result_list = {'train_loss' : [], 'train_acc' :[]}
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
        
        loss_sum = loss_sum / cnt
        loss_= round(loss_sum, 4)
        model.eval()
        acc = eval(model, test_loader)
        logger.info("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.total_epochs, acc))
        print("Epoch %d/%d, Acc=%.4f"%(epoch+1, args.total_epochs, acc))
        
        # result_list['train_loss'].append(loss_)
        # result_list['train_acc'].append(acc)
        
        # data_frame = pd.DataFrame(data=result_list, index=range(0, epoch + 1))
        # if args.q == 0 :
        #     csv_name = args.directory + "/csv_cifar10/resnet20_e_%d.csv"%(args.total_epochs) # Quan 추가. 나중에 수정해야함
        # else :
        #     csv_name = args.directory + '/csv_cifar10/512x256_resnet20_ab_%d_wb_%d_e_%d.csv'%(args.ab, args.wb, args.total_epochs)
        # data_frame.to_csv(csv_name, index_label='epoch')        

        if best_acc<acc:
            if args.q == 0 :
                torch.save( model, args.directory +'/ckpt_cifar10/resnet20_e_%d_round%d.pth'%(args.total_epochs, args.round) )
                if epoch <= 120 :
                    torch.save( model, args.directory +'/ckpt_cifar10/resnet20_e_120_round%d.pth'%(args.round))
            else :
                if args.mode=='train':
                    torch.save( model, args.directory +'/ckpt_cifar10/resnet20_ab_%d_wb_%d_e_%d_round%d.pth'%(args.ab, args.wb, args.total_epochs, args.round) )
                    if epoch <= 120 :
                        torch.save( model, args.directory +'/ckpt_cifar10/resnet20_ab_%d_wb_%d_e_120_round%d.pth'%(args.ab, args.wb, args.round))
                elif args.mode='prune' :
                    torch.save( model, args.directory +'/ckpt_cifar10/prune.pth')

            print('Model saved!!')
            best_acc=acc
        scheduler.step() # 스케쥴러 껐음

    acc_list.append(best_acc)
    logger.info("Best Acc=%.4f"%(best_acc))
    print("Best Acc=%.4f"%(best_acc))


def prune_model(model, block_pruned, block_pruned1):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) ) 
    def prune_conv(conv, to_p_amount=2): # 원래 0.2
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, to_p_amount=to_p_amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    # block_prune_probs = 0.5


    blk_id = 0
    # block_pruned = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock_Q) :
            prune_conv(m.conv0, block_pruned[blk_id])
            prune_conv(m.conv1, block_pruned1[blk_id])
            blk_id+=1
    return model    


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        if args.q == 0 :
            model = resnet20(num_classes=10)
        else :
            model = resnet20q(args.ab, args.wb, num_classes=10)
        print(model)
        logger.info(model)
        train_model(model, train_loader, test_loader)

    elif args.mode=='prune':
        previous_ckpt_ori = prune_root + '/resnet20_ab_%d_wb_%d_e_120_round%d.pth'%(args.ab, args.wb, args.round-1) 
        model_ori = torch.load(previous_ckpt_ori)
        acc_ori = eval(model_ori, test_loader)
        logger.info("Accuacy of the original network = %.4f\n"%(acc_ori))

        if args.q == 0 :
            previous_ckpt = prune_root + '/resnet20_e_120_round%d.pth'%(args.round-1) 
        else : 
            previous_ckpt = prune_root + '/resnet20_ab_%d_wb_%d_e_120_round%d.pth'%(args.ab, args.wb, args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        
        images = [32, 32, 32, 16, 16, 16, 8, 8, 8] # 원래 마지막에는 2임
        IC = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        array = [512, 256] # row, col
        bit_precision = 2
        memory_precision = 1
        block_pruned = []
        block_pruned1 = []
        tiled_ic = []
        after_ic = []
        cnt = 0
        for i in range(args.f) :
            print(cnt)
            block_pruned = []
            block_pruned1 = []
            if cnt == 0 :
                cnt = 1
                for i in range(len(images)) :
                    _, _, _, _, _, t_IC, _ = vw_sdk(images[i], images[i], 3, 3, IC[i], IC[i], array[0], array[1], 1, memory_precision, bit_precision)
                    if IC[i] < t_IC :
                        res_block = 0
                    else :
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
                    if i%3 != 1 :
                        block_pruned1[i] = 0
                cnt += 1

            else :
                previous_ckpt = prune_root + '/prune.pth' 
                print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
                for i in range(len(images)) :
                    _, _, _, _, _, t_IC, _= vw_sdk(images[i], images[i], 3, 3, after_ic[i], after_ic[i], array[0], array[1], 1, memory_precision, bit_precision)
                    res_block = after_ic[i] % t_IC
                    after_ic[i] = int(after_ic[i]) - res_block
                    block_pruned.append(res_block)
                    block_pruned1.append(res_block)
                    tiled_ic[i] = t_IC
                for i in range(len(block_pruned)-1) :
                    if after_ic[i] > after_ic[i+1] :
                        after_ic[i] = after_ic[i] - tiled_ic[i]
                        block_pruned[i] = block_pruned[i] + tiled_ic[i]

                for i in range(len(block_pruned1)) :
                    if i%3 != 1 :
                        block_pruned1[i] = 0

            print("block_pruned")
            print(block_pruned)
            print(block_pruned1)
            print("tiled_ic")
            print(tiled_ic)
            print("after")
            print(after_ic)

            block_pruned.append(0) # for last layer


            prune_model(model, block_pruned, block_pruned1)
            logger.info(model)
            params = sum([np.prod(p.size()) for p in model.parameters()])
            print("Number of Parameters: %.1fM"%(params/1e6))
            train_model(model, train_loader, test_loader)

    elif args.mode=='test':
        if args.q == 0 :
            ckpt = prune_root + '/resnet20_e360_round%d.pth'%(args.round)  # args.round
        else :
            ckpt = prune_root + '/resnet20_ab_%d_wb_%d_e360_round%d.pth'%(args.ab, args.wb, args.round) # args.round
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
