import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, output_channels, inter_channels, activation=True, se=False):
        super(BottleneckBlock, self).__init__()
        self.activation = activation
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(inter_channels, output_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        
        # Initialize bn3 gamma to zeros
        nn.init.zeros_(self.bn3.weight)
        
        if se:
            self.se = SqueezeExcitation(output_channels)
        else:
            self.se = None
            
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        # Third conv block
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se is not None:
            out = self.se(out)
        
        # Residual connection
        out = out + x
        
        if self.activation:
            out = F.relu(out, inplace=True)
            
        return out


class BottleneckIncBlock(nn.Module):
    def __init__(self, in_channels, output_channels, inter_channels, stride1x1=1, stride2x2=2, se=False):
        super(BottleneckIncBlock, self).__init__()
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, stride=stride1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        
        # 3x3 conv  
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, 3, stride=stride2x2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(inter_channels, output_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        
        # Initialize bn3 gamma to zeros
        nn.init.zeros_(self.bn3.weight)
        
        if se:
            self.se = SqueezeExcitation(output_channels)
        else:
            self.se = None
            
        # Shortcut connection
        stride = stride1x1 * stride2x2
        self.shortcut = nn.Conv2d(in_channels, output_channels, 1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se is not None:
            out = self.se(out)
        
        # Shortcut connection
        shortcut = self.shortcut(x)
        shortcut = self.shortcut_bn(shortcut)
        
        # Add and activate
        out = out + shortcut
        out = F.relu(out, inplace=True)
        
        return out


class SqueezeExcitation(nn.Module):
    """Squeeze-Excitation layer from https://arxiv.org/abs/1709.01507"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        out = self.avg_pool(x).view(b, c)
        # Excitation
        out = F.relu(self.fc1(out), inplace=True)
        out = torch.sigmoid(self.fc2(out)).view(b, c, 1, 1)
        # Scale
        return x * out.expand_as(x)


class ResNet(nn.Module):
    def __init__(self, filter_sizes, repeat_sizes, output_dim, final_pool=True, se=False):
        super(ResNet, self).__init__()
        assert len(filter_sizes) == len(repeat_sizes)
        
        self.initial_conv = nn.Conv2d(3, filter_sizes[0] // 4, 7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(filter_sizes[0] // 4)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Build residual blocks
        self.layers = nn.ModuleList()
        in_channels = filter_sizes[0] // 4
        
        for i in range(len(repeat_sizes)):
            out_channels = filter_sizes[i]
            inter_channels = filter_sizes[i] // 4
            
            # First block of the layer (with potential downsampling)
            stride = 2 if i > 0 else 1
            self.layers.append(
                BottleneckIncBlock(
                    in_channels, out_channels, inter_channels, 
                    stride1x1=1, stride2x2=stride, se=se
                )
            )
            
            # Remaining blocks
            for _ in range(repeat_sizes[i]):
                self.layers.append(
                    BottleneckBlock(
                        out_channels, out_channels, inter_channels, se=se
                    )
                )
            
            in_channels = out_channels
        
        self.final_pool = final_pool
        if final_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(filter_sizes[-1], output_dim)
        else:
            # Use conv + global average pooling
            self.conv_gap = nn.Conv2d(filter_sizes[-1], output_dim, 1, bias=False)
            self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # Residual blocks
        for layer in self.layers:
            x = layer(x)
        
        if self.final_pool:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.conv_gap(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
        
        x = self.sigmoid(x)
        return x


def create_resnet_152(input_shape, output_dim):
    """Original ResNet-152 Model."""
    filter_sizes = [256, 512, 1024, 2048]
    repeat_sizes = [2, 7, 35, 2]
    return ResNet(filter_sizes, repeat_sizes, output_dim, final_pool=True)


def create_resnet_custom_v1(input_shape, output_dim):
    """
    DeepDanbooru web (until 2019/04/20)
    Short, wide
    """
    filter_sizes = [256, 512, 1024, 2048, 4096]
    repeat_sizes = [2, 7, 35, 2, 2]
    return ResNet(filter_sizes, repeat_sizes, output_dim, final_pool=False)


def create_resnet_custom_v2(input_shape, output_dim):
    """
    Experimental (blazing-deep network)
    Deep, narrow
    """
    filter_sizes = [256, 512, 1024, 1024, 1024, 2048]
    repeat_sizes = [2, 7, 40, 16, 16, 6]
    return ResNet(filter_sizes, repeat_sizes, output_dim, final_pool=False)


def create_resnet_custom_v3(input_shape, output_dim):
    filter_sizes = [256, 512, 1024, 1024, 2048, 4096]
    repeat_sizes = [2, 7, 19, 19, 2, 2]
    return ResNet(filter_sizes, repeat_sizes, output_dim, final_pool=False)


def create_resnet_custom_v4(input_shape, output_dim):
    filter_sizes = [256, 512, 1024, 1024, 1024, 2048]
    repeat_sizes = [2, 7, 10, 10, 10, 2]
    return ResNet(filter_sizes, repeat_sizes, output_dim, final_pool=False)