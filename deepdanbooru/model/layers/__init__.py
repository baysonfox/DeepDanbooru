import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(
    in_channels, out_channels, kernel_size, stride=1, padding="same", initializer="he_normal"
):
    """Create a conv2d layer equivalent to TensorFlow version"""
    if padding == "same":
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )
    
    # Initialize weights using He normal initialization (equivalent to TF's he_normal)
    if initializer == "he_normal":
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')
    
    return conv_layer


class ConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        initializer="he_normal",
        bn_gamma_initializer="ones",
    ):
        super(ConvBN, self).__init__()
        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            initializer=initializer,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Initialize BatchNorm gamma (weight) parameter
        if bn_gamma_initializer == "ones":
            nn.init.ones_(self.bn.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding="same",
    initializer="he_normal",
    bn_gamma_initializer="ones",
):
    """Factory function for ConvBN layer"""
    return ConvBN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        initializer=initializer,
        bn_gamma_initializer=bn_gamma_initializer,
    )


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        initializer="he_normal",
        bn_gamma_initializer="ones",
    ):
        super(ConvBNReLU, self).__init__()
        self.conv_bn = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            initializer=initializer,
            bn_gamma_initializer=bn_gamma_initializer,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_bn(x)
        x = self.relu(x)
        return x


def conv_bn_relu(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding="same",
    initializer="he_normal",
    bn_gamma_initializer="ones",
):
    """Factory function for ConvBNReLU layer"""
    return ConvBNReLU(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        initializer=initializer,
        bn_gamma_initializer=bn_gamma_initializer,
    )


class ConvGAP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ConvGAP, self).__init__()
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = x.flatten(1)  # Flatten spatial dimensions
        return x


def conv_gap(in_channels, out_channels, kernel_size=1):
    """Factory function for ConvGAP layer"""
    return ConvGAP(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)


def repeat_blocks(x, block_delegate, count, **kwargs):
    """Apply a block function multiple times"""
    assert count >= 0

    for _ in range(count):
        x = block_delegate(x, **kwargs)
    return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation layer from https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        s = self.avg_pool(x).view(b, c)
        # Excitation
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s).view(b, c, 1, 1)
        # Scale
        return x * s.expand_as(x)


def squeeze_excitation(channels, reduction=16):
    """Factory function for SqueezeExcitation layer"""
    return SqueezeExcitation(channels=channels, reduction=reduction)
