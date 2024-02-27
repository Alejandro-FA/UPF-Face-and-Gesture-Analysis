import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

"""
Modified architecture of the original LightCNN architecture:
https://github.com/AlfredXiangWu/LightCNN/tree/master

If not done yet, read the comments at the beginning of the light_cnn.py file.

Modifications done to the LightCNN architecture to obtain the SuperLightCNN architecture:
    - Reduced the number of channels of the convolutiona layers.
    - Reduced output parameters of the first fully connected layers.
    - Removed one convolutional layer from the beginning of the architecture.

These modifications make the model adjust to the requirements imposed in our assignment:
    
    - Model size: The maximum size of the code and associated model files must
    not exceed 80 MB. ACCOMPLISHED by this model

    - Model depth: The maximum number of layers in deep learning-based models
    must not exceed 10 layers and 2 submodels. ACCOMPLISHED by this model

    - Model parameters: The model must contain less than 1 million parameters. ACCOMPLISHED by this model
"""


class mfm_v2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=1, groups: int=1, type: Literal[0,1]=1, batch_norm: bool = False):
        super(mfm_v2, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.bn = nn.BatchNorm2d(2*out_channels, momentum=0.1, affine=True) if batch_norm else None
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
            self.bn = nn.BatchNorm1d(2*out_channels, momentum=0.1, affine=True) if batch_norm else None

    def forward(self, x):
        out = self.filter(x)
        out = self.bn(out) if self.bn is not None else out
        out = torch.split(out, self.out_channels, 1)
        return torch.max(out[0], out[1])



class depth_wise_separable_mfm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, batch_norm: bool = False):
        super(depth_wise_separable_mfm, self).__init__()
        padding = kernel_size // 2
        self.conv_a = mfm_v2(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, batch_norm=batch_norm)
        self.conv_b = mfm_v2(in_channels, out_channels, kernel_size=1, padding=0, groups=1, batch_norm=batch_norm)

    def forward(self, x):
        out = self.conv_a(x)
        out = self.conv_b(out + x) # Skip connection
        return out
    
    

class inception_mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_1=3, kernel_size_2=5, batch_norm: bool=False):
        super(inception_mfm, self).__init__()
        assert out_channels % 2 == 0, "out_channels must be divisible by 2"
        self.conv1 = mfm_v2(in_channels, out_channels // 2, kernel_size=kernel_size_1, padding=kernel_size_1//2, batch_norm=batch_norm)
        self.conv2 = mfm_v2(in_channels, out_channels // 2, kernel_size=kernel_size_2, padding=kernel_size_2//2, batch_norm=batch_norm)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return torch.cat([out1, out2], dim=1)

    

class superlight_cnn_v3(nn.Module):
    def __init__(self, num_classes=79077, input_channels=1, batch_norm: bool=False):
        super(superlight_cnn_v3, self).__init__()
        self.conv1 = inception_mfm(input_channels, 24, kernel_size_1=5, kernel_size_2=7, batch_norm=batch_norm)
        self.conv2 = inception_mfm(24, 42, kernel_size_1=3, kernel_size_2=5, batch_norm=batch_norm)
        self.conv3 = depth_wise_separable_mfm(42, 84, 3, batch_norm=batch_norm)
        self.conv4 = depth_wise_separable_mfm(84, 56, 3, batch_norm=batch_norm)
        self.conv5 = depth_wise_separable_mfm(56, 56, 3, batch_norm=batch_norm)

        self.fc1 = mfm_v2(8*8*56, 128, type=0, batch_norm=batch_norm)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # Classification
        x = F.dropout(x, training=self.training)
        return self.fc2(x)
