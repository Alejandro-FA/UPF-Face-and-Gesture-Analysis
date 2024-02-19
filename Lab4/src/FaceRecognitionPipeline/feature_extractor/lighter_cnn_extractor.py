from .feature_extractor import FeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import imageio.v2

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            # Depth-wise separable convolutions
            self.filter = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=2*out_channels, kernel_size=1, stride=stride, padding=0)
            )
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

class lighter_network_9layers(nn.Module):
    def __init__(self, num_classes=79077, input_channels=1):
        super(lighter_network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(input_channels, 24, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(24, 48, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(96, 64, 3, 1, 1),
            group(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8*8*64, 128, type=0)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training) # x is the feature vector before the softmax layer
        out = self.fc2(x)
        return out#, x
    


class LighterCNN(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.torch_transform = transforms.ToTensor()
    

    def __call__(self, image: imageio.v2.Array) -> int:
        tensor: torch.Tensor = self.torch_transform(image)
        return -1 # TODO: Implement this method


    def save(file_path: str) -> None:
        raise NotImplementedError("Implement save method!")