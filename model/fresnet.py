# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   fresnet.py
    Time:        2022/10/31 14:40:57
    Editor:      Figo
-----------------------------------
'''

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import sys
sys.path.append("model")
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from robustness.tools.custom_modules import SequentialWithArgs, FakeReLU


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, wave='sym17', mode='append'):
        super(ResNet, self).__init__()

        self.wave = wave
        self.DWT = DWTForward(J=1, wave = self.wave, mode='symmetric',Requirs_Grad=True).cuda()
        self.FDmode = mode

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]
        self.num_classes = num_classes
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(12, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        
        x = self.wavelets(x, self.FDmode)

        out0 = F.relu(self.bn1(self.conv1(x)))
        
        out1 = self.layer1(out0)
        
        out2 = self.layer2(out1)

        out3 = self.layer3(out2)
    
        out4 = self.layer4(out3, fake_relu=fake_relu)

        out = F.avg_pool2d(out4, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        
        layer_feature = {'layer1': out1, 'layer2': out2, 'layer3': out3, 'layer4': out4, "layer4-v2": pre_out, "layer5": final}
        
        if with_latent:
            return final, pre_out, layer_feature
        return final
    
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out
    
    def wavelets(self, x, FDmode):
        # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',\n
        # 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'\n\n
        x = x.cuda().reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        Yl, Yh = self.DWT(x)
        output = self.plugdata(x, Yl, Yh, FDmode)
        return output
    
    def plugdata(self, x, Yl, Yh, mode):
        if mode == 'append':
            output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0:3, :] = Yl[:, :, :]
            output[:, 3:6, :] = Yh[0][:, 0, :, :]
            output[:, 6:9, :] = Yh[0][:, 1, :, :]
            output[:, 9:12, :] = Yh[0][:, 2, :, :]
            output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        elif mode == 'avg':
            output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
            output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
            output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
            output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
            output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        return output


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=5, **kwargs)

def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wd=.75, **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

resnet50 = ResNet50
resnet18 = ResNet18
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide

# resnet18thin = ResNet18Thin
def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


FRESNET = {
    "resnet18": resnet18, 
    "resnet50": resnet50, 
    "resnet101": resnet101, 
    "resnet152": resnet152, 
    "resnet18wide": resnet18wide
}