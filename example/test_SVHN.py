import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import SVHNQ, SVHN_
import cifar_resnet as resnet

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import SVHN
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
parser.add_argument('--batch_size', type=int, default=512) # 128로 줄이니까 돌아감
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=1)
parser.add_argument('--step_size', type=int, default=40) # 0.05일 떄, 75
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001) # 0.001 고정
parser.add_argument('--q', type=int, default=0) # 0 : fsull, 1 : quantize
parser.add_argument('--ab', type=int, default=0)
parser.add_argument('--wb', type=int, default=0)
parser.add_argument('--optim', type=str, default = 'Adam') # 아담 고정
parser.add_argument('--directory', type=str, default = '/Data') # fc layer 크기
parser.add_argument('--scheduler', type=str, default = 'Multi_StepLR')
parser.add_argument('--f', type=int, default = 20)

args = parser.parse_args()

if args.q == 0:
    file_name = './log_svhn2/svhn_e_%d.log'%(args.total_epochs)
else :
    # file_name = './log_svhn2/sort_256x128_1bit_svhn_ab_%d_wb_%d_e_%d.log'%(args.ab, args.wb, args.total_epochs)
    file_name = './log_svhn2/test128x512.log'
file_handler = logging.FileHandler(file_name)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

prune_root = '/Data/ckpt_svhn2' 

# 85 // 93

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        SVHN('/Data', transform=transforms.Compose([
            transforms.Resize(40),
            transforms.CenterCrop(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]), download=True, split='train'), shuffle=True, batch_size=args.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        SVHN('/Data', transform=transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]),download=True, split='test'), shuffle=False, batch_size=args.batch_size, num_workers=0)
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

    if args.optim == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else :    
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.scheduler == 'StepLR' :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1) # 스케쥴러 설정

    elif args.scheduler == 'Multi_StepLR' :
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 90], gamma=0.1) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1) 



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
        #     csv_name = args.directory + "/csv_svhn2/svhn_e_%d.csv"%(args.total_epochs) # Quan 추가. 나중에 수정해야함
        # else :
        #     csv_name = args.directory + '/csv_svhn2/sort_256x128_1bit_svhn_ab_%d_wb_%d_e_%d.csv'%(args.ab, args.wb, args.total_epochs)
        # data_frame.to_csv(csv_name, index_label='epoch')        

        if best_acc<acc:
            if args.q == 0 :
                torch.save( model, args.directory +'/ckpt_svhn2/svhn_e_%d_round%d.pth'%(args.total_epochs, args.round) )
                if epoch <= 60 :
                    torch.save( model, args.directory +'/ckpt_svhn2/svhn_e_60_round%d.pth'%(args.round))
            else :
                if args.mode=='train':
                    torch.save( model, args.directory +'/ckpt_svhn2/svhn_ab_%d_wb_%d_e_%d_round%d.pth'%(args.ab, args.wb, args.total_epochs, args.round) )
                    if epoch <= 60 :
                        torch.save( model, args.directory +'/ckpt_svhn2/svhn_ab_%d_wb_%d_e_60_round%d.pth'%(args.ab, args.wb, args.round))       
                elif args.mode=='prune':
                    torch.save( model, args.directory +'/ckpt_svhn2/prune.pth')

            print('Model saved!!')
            best_acc=acc
        scheduler.step() 

    acc_list.append(best_acc)
    logger.info("Best Acc=%.4f"%(best_acc))
    print("Best Acc=%.4f"%(best_acc))

    return best_acc



def prune_model(model, block_pruned):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 40, 40))
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

    blk_id = 0
    # block_pruned = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for m in model.modules():
        if isinstance(m, nn.Conv2d) :
            prune_conv(m, block_pruned[blk_id])
            blk_id+=1
    return model    


def main():
    train_loader, test_loader = get_dataloader()

    if args.mode=='train':
        args.round=0
        if args.q == 0 :
            model = SVHN_().cuda()
        else :
            model = SVHNQ(args.ab, args.wb, num_classes=10).cuda()
        print(model)
        logger.info(model)
        train_model(model, train_loader, test_loader)


    elif args.mode=='prune':
        previous_ckpt_ori = prune_root + '/svhn_ab_%d_wb_%d_e_100_round%d.pth'%(args.ab, args.wb, args.round-1) 
        model_ori = torch.load(previous_ckpt_ori)
        acc_ori = eval(model_ori, test_loader)
        logger.info("Accuacy of the original network = %.4f\n"%(acc_ori))

        if args.q == 0 :
            previous_ckpt = prune_root + '/svhn_e_60_round%d.pth'%(args.round-1) 
        else : 
            previous_ckpt = prune_root + '/svhn_ab_%d_wb_%d_e_60_round%d.pth'%(args.ab, args.wb, args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load(previous_ckpt)

        '''
        VGG13 information
        '''

        images = [18, 18, 9, 7, 7, 5] # 원래 마지막에는 2임
        IC = [24, 32, 32, 64, 64, 64]
        OC = [32, 32, 64, 64, 64, 256]
        K = [3, 3, 3, 3, 3, 5]
        array = [256, 512] # row, col // Deep Neural Network Acceleration in Non-Volatile Memory: A Digital Approach
        bit_precision = 1
        memory_precision = 1
        after_ic = []
        after_oc = []
        tiled_ic = []
        cnt = 0
        for i in range(args.f) :
            block_pruned = []
            if cnt == 0 :
                for i in range(len(images)) :
                    _, tc, rr, rc, _, t_IC, _ = vw_sdk(images[i], images[i], K[i], K[i], IC[i], OC[i], array[0], array[1], 1, memory_precision, bit_precision)
                    if IC[i] < t_IC :
                        res_block = 0
                    else :
                        res_block = IC[i] % t_IC

                    after_ic.append(int(IC[i]) - res_block)
                    block_pruned.append(res_block)
                    tiled_ic.append(t_IC)

                for i in range(len(block_pruned)-1) : # 점점 layer가 커진다는 것을 가정하면.
                    t = len(block_pruned) - i - 1
                    if after_ic[t-1] > after_ic[t] :
                        after_ic[t-1] = after_ic[t-1] - tiled_ic[t-1]
                        block_pruned[t-1] = block_pruned[t-1] + tiled_ic[t-1]
                
                for i in range(len(after_ic)-1) :
                    after_oc.append(after_ic[i+1])
                after_oc.append(64)
                cnt += 1

            else :
                previous_ckpt = prune_root + '/prune.pth' 
                print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
                model = torch.load(previous_ckpt)
                for i in range(len(images)) :
                    _, over_col, over_row, _, _, t_IC, _ = vw_sdk(images[i], images[i], K[i], K[i], after_ic[i], after_oc[i], array[0], array[1], 1, memory_precision, bit_precision)
                    print(t_IC, over_row, over_col)
                    if after_ic[i] < t_IC :
                        res_block = 0
                    else :
                        res_block = after_ic[i] % t_IC

                    after_ic[i] = int(after_ic[i]) - res_block
                    tiled_ic[i] = t_IC
                    block_pruned.append(res_block)

                for i in range(len(block_pruned)-1) : # 점점 layer가 커진다는 것을 가정하면.
                    t = len(block_pruned) - i - 1
                    if after_ic[t-1] > after_ic[t] :
                        after_ic[t-1] = after_ic[t-1] - tiled_ic[t-1]
                        block_pruned[t-1] = block_pruned[t-1] + tiled_ic[t-1]
                
                for i in range(len(after_ic)-1) :
                    after_oc[i] = after_ic[i+1]

            print(after_ic)
            print(after_oc)
            print("block_pruned")
            print(block_pruned)
            print("tiled_IC")
            print(tiled_ic)
            print("after IC")
            print(after_ic)
            print("after OC")
            print(after_oc)

            block_pruned.append(0) # for last layer
            prune_model(model, block_pruned)
            print(model)
            logger.info(model)
            params = sum([np.prod(p.size()) for p in model.parameters()])
            print("Number of Parameters: %.1fM"%(params/1e6))
            logger.info(params)
            acc_p = train_model(model.cuda(), train_loader, test_loader)
            logger.info("block pruned = " + str(block_pruned))
            logger.info("Sum of block pruned = %.1f\n"%(sum(block_pruned)))
            logger.info("Accuacy of the pruned network = %.4f\n"%(acc_p))
            logger.info("Accuacy gap = %.4f\n"%(acc_ori - acc_p))
            # if (sum(block_pruned)==0) or (acc_ori - acc_p > 0.01):
            # if sum(block_pruned)==0 :
            #     break
            # if i != 0 :
            #     if acc_list[i-1] > acc_list[i] :
            #         break

    elif args.mode=='test':
        if args.q == 0 :
            ckpt = prune_root + '/svhn_e_60_round%d.pth'%(args.round)  # args.round
        else :
            ckpt = prune_root + '/svhn_ab_%d_wb_%d_e_60_round%d.pth'%(args.ab, args.wb, args.round) # args.round
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
