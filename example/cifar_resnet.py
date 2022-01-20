import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from utils import *
 
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.act = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.act0 = nn.ReLU()
        self.conv0 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = self.bn0(x)
        out = self.act0(out)
        out = self.bn1(self.conv0(x))
        out = self.act1(out)
        out = self.conv1(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        self.act = nn.ReLU()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        # 32
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) 
        # out = self.layer4(out)
        # 4
        out = F.avg_pool2d(out, 8)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature

class BasicBlock_Q(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, a_bit, w_bit, stride=1):
        super(BasicBlock_Q, self).__init__()
        
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.act = Activate(self.a_bit)

        self.act0 = Activate(self.a_bit)
        self.conv0 = Conv2d_Q_(self.w_bit, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = SwitchBatchNorm2d_(self.w_bit, planes)
        self.act1 = Activate(self.a_bit)
        self.conv1 = Conv2d_Q_(self.w_bit, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SwitchBatchNorm2d_(self.w_bit, planes)

        self.skip_conv = None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential( # 여기서는 quantization
                # Conv2d_Q_(self.w_bit, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                # SwitchBatchNorm2d_(self.w_bit, self.expansion*planes)
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = self.act0(x)
        out = self.bn0(self.conv0(out))
        out = self.act1(out)
        out = self.conv1(out)
        out += self.shortcut(x)
        out = self.bn1(out)
        return out
 

class ResNet_Q(nn.Module):
    def __init__(self, block, num_blocks, a_bit, w_bit, num_classes=10):
        super(ResNet_Q, self).__init__()
        
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.act = Activate(self.a_bit)

        self.in_planes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SwitchBatchNorm2d_(self.w_bit, 64)
        self.layer1 = self._make_layer_(block, 64, num_blocks[0], self.a_bit, self.w_bit, stride=1)
        self.layer2 = self._make_layer_(block, 128, num_blocks[1], self.a_bit, self.w_bit, stride=2)
        self.layer3 = self._make_layer_(block, 256, num_blocks[2], self.a_bit, self.w_bit, stride=2)
        # self.layer4 = self._make_layer_(block, 512, num_blocks[3], self.a_bit, self.w_bit, stride=2)
        # self.linear = Linear_Q_(self.w_bit, 512*block.expansion, num_classes)
        self.linear = nn.Linear(256*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer_(self, block, planes, num_blocks, a_bit, w_bit, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # print(strides)
        layers = []
        for stride in strides:
            # print(stride)
            layers.append(block(self.in_planes, planes, a_bit, w_bit, stride))
            self.in_planes = planes * block.expansion
        # print(layers)
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature
 

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet18_Q(a_bit, w_bit, num_classes=10) :
    return ResNet_Q(BasicBlock_Q, [2,2,2,2], a_bit, w_bit, num_classes)

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

def resnet20q(a_bit, w_bit, num_classes=10):
    return ResNet_Q(BasicBlock_Q, [3, 3, 3], a_bit, w_bit, num_classes)

class VGG13(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG13, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False) # 32
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1 , bias=False) # 42
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False) # 84
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False) # 112
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False) # 224
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False) # 224
        self.bn3 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False) # 504
        self.bn4 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 504
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 504
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False) # 504

        # self.prelu = nn.PReLU()
        # self.relu = F.relu()
        
        self.dropout = nn.Dropout(p=0.5)
        #512 7 4
        self.avg_pool = nn.AvgPool2d(2) # 7 // 2
        
        #512 1 1
        self.fc1 = nn.Linear(512*1*1, 512, bias=False)
        self.fc2 = nn.Linear(512, 512, bias=False) # 1024, 128
        self.classifier = nn.Linear(512, 10, bias=False) # 128


    def forward(self, x, out_feature=False) :
        # x = F.relu(self.conv1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.maxpool(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.maxpool(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)

        features = x
        # print(features.shape)
        # x = self.avg_pool(features)
        x = x.view(features.size(0), -1)
        x = self.dropout(F.relu(x)) # relu 없애도 될듯?
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc2(x)
        x = self.classifier(x)

        if out_feature == False:
            return x
        else:
            return x, features

class VGG13_Q(nn.Module):
    def __init__(self, a_bit, w_bit, num_classes=10):
        super(VGG13_Q, self).__init__()
        
        self.a_bit = a_bit
        self.w_bit = w_bit

        # Conv2d = conv2d_quantize_fn(self.w_bit)
        # Linear = linear_quantize_fn(self.w_bit)
        # batchnorm = batchnorm2d_fn(self.a_bit)
        self.act = Activate(self.a_bit)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2 = Conv2d_Q_(self.w_bit, 64, 64, kernel_size=3, padding=1, stride=1 , bias=False)
        self.conv3 = Conv2d_Q_(self.w_bit, 64, 128, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv4 = Conv2d_Q_(self.w_bit, 128, 128, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv5 = Conv2d_Q_(self.w_bit, 128, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv6 = Conv2d_Q_(self.w_bit, 256, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv7 = Conv2d_Q_(self.w_bit, 256, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv8 = Conv2d_Q_(self.w_bit, 512, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv9 = Conv2d_Q_(self.w_bit, 512, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv10 = Conv2d_Q_(self.w_bit, 512, 512, kernel_size=3, padding=1, stride=1, bias=False)

        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = SwitchBatchNorm2d_(self.w_bit, 64)
        self.bn2 = SwitchBatchNorm2d_(self.w_bit, 128)
        self.bn3 = SwitchBatchNorm2d_(self.w_bit, 256)
        self.bn4 = SwitchBatchNorm2d_(self.w_bit, 512)
        self.fc1 = Linear_Q_(self.w_bit, 512*1*1, 512, bias=False)
        self.fc2 = Linear_Q_(self.w_bit, 512,512, bias=False) 
        self.classifier = nn.Linear(512,10, bias=True) # 마지막 layer는 quantization 제외

    def forward(self, x, out_feature=False) :
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.maxpool(x) 

        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        x = self.maxpool(x)
        
        x = self.conv5(x)
        x = self.act(x)
        x = self.conv6(x)
        x = self.act(x)
        x = self.maxpool(x)      

        x = self.conv7(x)
        x = self.act(x)
        x = self.conv8(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.conv9(x)
        x = self.act(x)
        x = self.conv10(x)
        x = self.act(x)
        x = self.maxpool(x)

        features = x
        x = x.view(features.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.classifier(x)

        if out_feature == False:
            return x
        else:
            return x, features

class SVHNQ(nn.Module):
    def __init__(self, a_bit, w_bit, num_classes=10, expand=2): # 원래 expand = 2 // 8도 테스트 해볼것
        super(SVHNQ, self).__init__()
        self.w_bit = w_bit
        self.a_bit = a_bit
        # nn.Upsampling(...)
        self.act = Activate(self.a_bit)

        self.expand = expand

        ep = self.expand
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=0, bias=True), # ep*6
            nn.MaxPool2d(2),
            Activate(self.a_bit),
            # 18
            Conv2d_Q_(self.w_bit, 24, 32, kernel_size=3, padding=1, stride=1 , bias=False),
            SwitchBatchNorm2d_(self.w_bit, 32),
            Activate(self.a_bit),

            Conv2d_Q_(self.w_bit, 32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d_(self.w_bit, 32),
            nn.MaxPool2d(2),
            Activate(self.a_bit),
            # 9
            Conv2d_Q_(self.w_bit, 32, 64, kernel_size=3, padding=0, stride=1, bias=False),
            SwitchBatchNorm2d_(self.w_bit, 64),
            Activate(self.a_bit),
            # 7
            Conv2d_Q_(self.w_bit, 64, 64, 3, padding=1, stride=1, bias=False),
            SwitchBatchNorm2d_(self.w_bit, 64),
            Activate(self.a_bit),

            Conv2d_Q_(self.w_bit, 64, 64, 3, padding=0, stride=1, bias=False),
            SwitchBatchNorm2d_(self.w_bit, 64),
            Activate(self.a_bit),
            # 5
            nn.Dropout(0.5),
            Conv2d_Q_(self.w_bit, 64, 256, 5, padding=0, stride=1, bias=False),
            Activate(self.a_bit, quantize=False),
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view([x.shape[0], -1])
        x = self.fc(x)
        return x

class SVHN_(nn.Module):
    def __init__(self, expand=4): # 원래 expand = 2 // 8도 테스트 해볼것
        super(SVHN_, self).__init__()
        

        self.expand = expand

        ep = self.expand
        self.layers = nn.Sequential(
            nn.Conv2d(3, ep * 6, 5, padding=0, bias=True),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 18
            nn.Conv2d(ep*6, ep*8, kernel_size=3, padding=1, stride=1 , bias=False),
            nn.BatchNorm2d(ep*8),
            nn.ReLU(),
            nn.Conv2d(ep * 8, ep * 8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(ep * 8),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 9
            nn.Conv2d( ep * 8, ep * 16, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ep * 16),
            nn.ReLU(),
            # 7
            nn.Conv2d(ep * 16, ep * 16, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(ep * 16),
            nn.ReLU(),
            nn.Conv2d(ep * 16, ep * 16, 3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ep * 16),
            nn.ReLU(),
            # 5
            nn.Dropout(0.5),
            nn.Conv2d(ep * 16, ep * 64, 5, padding=0, stride=1, bias=False),
            nn.ReLU(),
        )
        self.fc = nn.Linear(ep * 64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view([x.shape[0], -1])
        x = self.fc(x)
        return x
