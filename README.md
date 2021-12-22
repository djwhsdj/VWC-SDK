# VWC-SDK: Convolutional Weight Mapping Using Shifted and Duplicated Kernel with Variable Windows and Channels.
### This paper extends the prior work VW-SDK: Efficient Convolutional Weight Mapping Using Variable Windows for Processing-In-Memory Architecrues.
### The prior work is accepted as a conference papaer at Design, Automation & Test in Europe Conference & Exhibition (DATE) 2022. 
### You can read this prior paper pdf [here](https://arxiv.org/abs/2112.11282)
---
## Abstract
With their high energy efficiency, processing-in-memory (PIM) arrays are increasingly used for convolutional neural network (CNN) inference. In PIM-based CNN inference, the computational latency and energy are dependent on how the CNN weights are mapped to the PIM array. A recent study proposed shifted and duplicated kernel (SDK) mapping that reuses the input feature maps with a unit of a parallel window, which is convolved with duplicated kernels to obtain multiple output elements in parallel. However, the existing SDK-based mapping algorithm does not always result in the minimum computing cycles because it only maps a square-shaped parallel window with the entire channels. In this paper, we introduce a novel mapping algorithm called variable-window SDK (VW-
SDK), which adaptively determines the shape of the parallel window that leads to the minimum computing cycles for a given convolutional layer and PIM array. By allowing rectangular-shaped windows with partial channels, VW-SDK utilizes the PIM array more efficiently, thereby further reducing the number of computing cycles. However, when VW-SDK finds the optimal shape of the parallel window, the residual channel occurs, leading to the increase of the computing cycle. To address this problem, we propose an algorithm called variable windows and channels SDK (VWC-SDK) that integrates VW-SDK and channel pruning. By pruning some channels, VWC-SDK finds the optimal number of channels in each convolutional layer. The simulation with a 256×512 PIM array and CNN-8 shows that VWC-SDK improves the inference speed by 1.38×compared to the original network

## Requirements
+ python 3.6.9
+ pytorch 1.9.1
+ numpy

## Usage

### test_SVHN.py
* This code training and pruning the CNN-8 on SVHN.
* If you train this model, scheduler(milestones=[50, 75, 90])

command: python3 test_SVHN.py --mode train --q 1 --ab 1 --wb 1 --total_epochs 100 --lr 0.001

* If you prune this moel, you have to change your directory and scheduler(milestones=[10, 20, 30])

command: python3 test_SVHN.py --mode prune --q 1 --ab 1 --wb 1 --total_epochs 40 --lr 0.001

### test_resnet20_cp.py
* This code training and pruning the Resnet-20 on cifar10.
* If you train this model, scheduler(milestones=[75, 125, 175])

command: python3 test_resnet20_cp.py --mode train --q 1 --ab 2 --wb 2 --total_epochs 200 --lr 0.001

* If you prune this moel, you have to change your directory and scheduler(milestones=[30, 50, 70])

command: python3 test_resnet20_cp.py --mode prune --q 1 --ab 1 --wb 1 --total_epochs 80 --lr 0.001

### cifar_resnet.py
* There are the networks including the original and quantized model.

### utils.py
* There are the quantization function and modified VW-SDK.


## Results
### 1) CNN-8 on SVHN, where the PIM array size is 256x512, and the wieght bit precision is 1-bit.
| **Conv** | **IC** | **OC** | **Params** | **<td colspan=4>VW-SDK** | **<td colspan=4>RCP** |
|:---:|:---:|---:|---:|---:|:---:|
| | | | | 13x6 | - | - | - | 3 | 18 | 1350 | - | 
| 224x224 | 3x3x3x64 | 49284 | 12321 | 6216 | 10x3x3x64 |
| 224x224 | 3x3x64x64 | 98568 | 24642 | 24642 | 4x4x32x128 |
| 112x112 | 3x3x64x128 | 24200 | 6050 | 6050 | 4x4x32x128 |
| 112x112 | 3x3x128x128 | 36300 | 36300 | 12100 | 4x4x32x128 |
| 56x56 | 3x3x128x256 | 8748 | 8748 | 5832 | 4x3x42x256 |
| 56x56 | 3x3x256x256 | 14580 | 14580 | 10206 | 4x3x42x256 |

# Reference paper and codes
### Any-precision DNN
You can read the original pdf [here](https://arxiv.org/abs/1911.07346), and code [here](https://github.com/SHI-Labs/Any-Precision-DNNs) 

### Pruning filters for efficient convnets
You can read the original pdf [here](https://arxiv.org/pdf/1608.08710.pdf), and code [here](https://github.com/VainF/Torch-Pruning) 
