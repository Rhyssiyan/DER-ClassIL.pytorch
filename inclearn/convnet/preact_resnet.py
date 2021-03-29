import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, remove_last_relu=False):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.remove_last_relu = remove_last_relu

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        out = self.bn3(out)
        if not self.remove_last_relu:
            out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 nf=64,
                 zero_init_residual=True,
                 dataset="cifar",
                 start_class=0,
                 remove_last_relu=False):
        super(PreActResNet, self).__init__()
        self.in_planes = nf
        self.dataset = dataset
        self.remove_last_relu = remove_last_relu

        if 'cifar' in dataset:
            self.conv1 = nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, num_blocks[3], stride=2, remove_last_relu=remove_last_relu)
        self.out_dim = 8 * nf

        if 'cifar' in dataset:
            self.avgpool = nn.AvgPool2d(4)
        elif 'imagenet' in dataset:
            self.avgpool = nn.AvgPool2d(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ---------------------------------------------
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, PreActBlock):
        #             nn.init.constant_(m.bn2.weight, 0)
        #         elif isinstance(m, PreActBottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        # ---------------------------------------------

    def _make_layer(self, block, planes, num_blocks, stride, remove_last_relu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if remove_last_relu:
            for i in range(len(strides) - 1):
                layers.append(block(self.in_planes, planes, strides[i]))
                self.in_planes = planes * block.expansion
            layers.append(block(self.in_planes, planes, strides[-1], remove_last_relu=True))
            self.in_planes = planes * block.expansion
        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], **kwargs)


def PreActResNet34(**kwargs):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], **kwargs)


def PreActResNet50(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], **kwargs)


def PreActResNet101(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], **kwargs)


def PreActResNet152(**kwargs):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], **kwargs)
