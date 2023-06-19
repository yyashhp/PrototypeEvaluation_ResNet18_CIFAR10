import torch
import torch.nn as nn
import torch.nn.functional as F



class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, HW, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, nclass = 10, scale = 1, channels = 3, block_type = Block, **kwargs):
        super(ResNet, self).__init__()


        self.in_channels = int(64 * scale)
        self.orig_channels = int(64 * scale)
        print(f"Printing orig channels: {self.orig_channels}")
        self.orig_HW = 32
        self.channels = channels
        print(self.in_channels)

        self.conv1 = nn.Conv2d(self.channels, self.orig_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self._make_layer(self.orig_channels, num_blocks, HW=self.orig_HW, stride=1)
        self.layer2 = self._make_layer(self.orig_channels * 2, num_blocks, HW=(0.5 * self.orig_HW), stride=2)
        self.layer3 = self._make_layer(self.orig_channels * 4, num_blocks, HW=(0.25 * self.orig_HW), stride=2)
        self.layer4 = self._make_layer(self.orig_channels * 8, num_blocks, HW=(0.125 * self.orig_HW), stride=2)
        self.linear = nn.Linear(self.in_channels * block_type.expansion, nclass)

        self.multi_out = 0
        #self.proto_layer = kwargs['proto_layer']
        #self.proto_norm = kwargs['proto_norm']
        #self.proto_pool = "ave"
        #self.proto_pool_f = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, num_blocks, HW, stride, block_type = Block):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block_type(self.in_channels, out_channels, HW, stride))
            self.in_channels = out_channels * block_type.expansion
        return nn.Sequential(*layers)

    #def define_proto(self, features):
    #    if self.proto_pool:
    #        features = self.proto_pool_f(features)
    #    return features.view(features.shape[0],-1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #if self.proto_layer == 3:
        #p = self.define_proto(out)
        out = F.avg_pool2d(out,4)

        p = out.view(out.size(0), -1)

       # if self.proto_norm:

        out = self.linear(p)

        if (self.multi_out):
            return p, out #returning feature weights as well as logits
        else:
            return out #just returning logits
def ResNet18(nclass, scale, channels, **kwargs):
    return ResNet(2, nclass, scale, channels, **kwargs)
