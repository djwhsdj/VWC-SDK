import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss
        
class Activate(nn.Module):
    def __init__(self, a_bit, quantize=True):
        super(Activate, self).__init__()
        self.abit = a_bit
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.abit)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        self.abit = a_bit
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn.apply(x, self.abit)
        return activation_q


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        self.wbit = w_bit
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            weight_q = weight_q * E
        return weight_q


class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, w_bit, num_features):
        super(SwitchBatchNorm2d, self).__init__()
        self.w_bit = w_bit
        self.bn_dict = nn.ModuleDict()
        # for i in self.bit_list:
        #     self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)
        self.bn_dict[str(w_bit)] = nn.BatchNorm2d(num_features)

        self.abit = self.w_bit
        self.wbit = self.w_bit
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x

class SwitchBatchNorm2d_(SwitchBatchNorm2d) : ## 만든거
    def __init__(self, w_bit, num_features) :
        super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)
        self.w_bit = w_bit      
        # return SwitchBatchNorm2d_
    


def batchnorm2d_fn(w_bit):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, w_bit=w_bit):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, w_bit=w_bit)

    return SwitchBatchNorm2d_


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)

class Conv2d_Q_(Conv2d_Q): ## 만든거
    def __init__(self, w_bit, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                    bias=True):
        super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                        bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight) # nn.Conv2d에서 오는 값인듯?
        # test = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # print(test)
        return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

# def conv2d_quantize_fn(w_bit):
#     class Conv2d_Q_(Conv2d_Q):
#         def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                      bias=True):
#             super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                             bias)
#             self.bit_list = w_bit
#             # self.bit_list = bit_list
#             # self.w_bit = self.bit_list[-1]
#             self.quantize_fn = weight_quantize_fn(self.bit_list)

#         def forward(self, input, order=None):
#             weight_q = self.quantize_fn(self.weight)
#             return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

#     return Conv2d_Q_


class Linear_Q(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Q, self).__init__(*kargs, **kwargs)

class Linear_Q_(Linear_Q): ## 만든거
    def __init__(self, w_bit, in_features, out_features, bias=True):
        super(Linear_Q_, self).__init__(in_features, out_features, bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(self.w_bit)

    def forward(self, input, order=None):
        weight_q = self.quantize_fn(self.weight)
        return F.linear(input, weight_q, self.bias)

# def linear_quantize_fn(w_bit):
#     class Linear_Q_(Linear_Q):
#         def __init__(self, in_features, out_features, bias=True):
#             super(Linear_Q_, self).__init__(in_features, out_features, bias)
#             self.w_bit = w_bit
#             self.quantize_fn = weight_quantize_fn(self.w_bit)

#         def forward(self, input):
#             weight_q = self.quantize_fn(self.weight)
#             return F.linear(input, weight_q, self.bias)

#     return Linear_Q_



def vw_sdk (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col, Ks, memory_precision, bit_precision) :
    
    real_used_col = math.floor(array_col * memory_precision / bit_precision)

    i = 0 # initialize # overlap col
    j = 1 # overlap row

    reg_total_cycle = [] # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    cnt = 1
    while True :
        try :
            i += 1
            if i + filter_col - 1 > image_col : 
                i = 1
                j += 1
                cnt += 1
                if j + filter_row - 1 > image_row : 
                    break
            
            pw_row = filter_row + i + Ks - 2
            pw_col = filter_col + j + Ks - 2
            PWs_w = pw_row - filter_row + Ks
            PWs_h = pw_col - filter_col + Ks

            reg_N_parallel_window_row = math.ceil((image_row - pw_row)/PWs_w) + 1
            reg_N_parallel_window_col = math.ceil((image_col - pw_col)/PWs_h) + 1

            
            # for cycle computing
            # Tiled IC
            if in_channel == 3 :
                ICt = math.floor(array_row /(filter_row+i-1)*(filter_col+j-1))
                if ICt > in_channel :
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else :
                ICt = math.floor(array_row /(pw_row*pw_col))
                row_cycle = math.ceil(in_channel / ICt)
            
            # Tiled OC
            OCt =  math.floor(real_used_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)
    
            reg_N_of_computing_cycle = reg_N_parallel_window_row * reg_N_parallel_window_col \
                                    * row_cycle * col_cycle
            
            if i == 1 : # initialize
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            if reg_total_cycle[0] > reg_N_of_computing_cycle :
                del reg_total_cycle[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]

                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

    
        except ZeroDivisionError :
            continue
    print(reg_overlap_row, reg_overlap_col, reg_total_cycle[0])
    return reg_total_cycle[0], reg_overlap_col[0], reg_overlap_row[0], reg_row_cycle[0], reg_col_cycle[0], reg_ICt[0], reg_OCt[0] 
