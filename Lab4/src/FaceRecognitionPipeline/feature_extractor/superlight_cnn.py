import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modified architecture of the original LightCNN architecture Original architecture:
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


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            
            # Depth-wise separable convolutions
            # self.filter = nn.Sequential(
            #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
            #     nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1, stride=stride, padding=0)
            # )
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])



class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x



class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out



class superlight_network_9layers(nn.Module):
    def __init__(self, num_classes=79077, input_channels=1):
        super(superlight_network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(input_channels, 16, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            mfm(16, 32, 3, 1, 1), 
            # group(16, 32, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(32, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(64, 48, 3, 1, 1),
            group(48, 48, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8*8*48, 128, type=0)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training) # x is the feature vector before the softmax layer
        out = self.fc2(x)
        return out#, x