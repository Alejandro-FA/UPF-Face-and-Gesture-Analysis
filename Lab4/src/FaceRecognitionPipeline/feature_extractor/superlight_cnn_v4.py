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


class mfm_v3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=1, type: Literal[0,1]=1, instance_norm: bool = False):
        super(mfm_v3, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.norm = nn.InstanceNorm2d(2*out_channels, momentum=0.1, affine=False) if instance_norm else None
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
            self.norm = None

    def forward(self, x):
        out = self.filter(x)
        out = self.norm(out) if self.norm is not None else out
        out = torch.split(out, self.out_channels, 1)
        return torch.max(out[0], out[1])

    

class group_v3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, instance_norm: bool):
        super(group_v3, self).__init__()
        self.conv_a = mfm_v3(in_channels, in_channels, 1, 1, 0, instance_norm=instance_norm)
        self.conv   = mfm_v3(in_channels, out_channels, kernel_size, stride, padding, instance_norm=instance_norm)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x
    
    

class inception_mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_1=3, kernel_size_2=5, instance_norm: bool=False):
        super(inception_mfm, self).__init__()
        assert out_channels % 2 == 0, "out_channels must be divisible by 2"
        self.conv1 = mfm_v3(in_channels, out_channels // 2, kernel_size=kernel_size_1, padding=kernel_size_1//2, instance_norm=instance_norm)
        self.conv2 = mfm_v3(in_channels, out_channels // 2, kernel_size=kernel_size_2, padding=kernel_size_2//2, instance_norm=instance_norm)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        return torch.cat([out1, out2], dim=1)

    

class superlight_cnn_v4(nn.Module):
    def __init__(self, num_classes=79077, input_channels=1, instance_norm: bool=False):
        super(superlight_cnn_v4, self).__init__()
        self.conv1 = inception_mfm(input_channels, 16, kernel_size_1=5, kernel_size_2=7, instance_norm=instance_norm)
        self.conv2 = inception_mfm(16, 32, kernel_size_1=3, kernel_size_2=5, instance_norm=instance_norm)
        self.conv3 = group_v3(32, 64, 3, 1, 1, instance_norm=instance_norm)
        self.conv4 = group_v3(64, 48, 3, 1, 1, instance_norm=instance_norm)
        self.conv5 = group_v3(48, 48, 3, 1, 1, instance_norm=instance_norm)

        self.fc1 = mfm_v3(8*8*48, 133, type=0, instance_norm=instance_norm)
        self.fc2 = nn.Linear(133, num_classes)

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
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc2(x)
