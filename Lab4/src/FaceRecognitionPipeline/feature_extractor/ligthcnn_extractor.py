from .feature_extractor import FeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import imageio.v2
import os
import matplotlib.pyplot as plt
from PIL import Image

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        print(x.shape)
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

class network_9layers(nn.Module):
    def __init__(self, num_classes=79077, input_channels=1):
        super(network_9layers, self).__init__()
        self.features = nn.Sequential(
            mfm(input_channels, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training) # x is the feature vector before the softmax layer
        out = self.fc2(x)
        return out#, x
    


class LightCNN(FeatureExtractor):
    def __init__(self, model_path: str, threshold=0.2, num_classes=80, input_channels=3) -> None:
        super().__init__()
        if not os.path.isfile(model_path):
            raise ValueError(f"Invalid file {model_path}")
        
        self.torch_transform = transforms.ToTensor()
        self.model = network_9layers(num_classes=num_classes, input_channels=input_channels)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.threshold = threshold
    

    def __call__(self, image: imageio.v2.Array) -> int:
        tensor: torch.Tensor = self.torch_transform(image).unsqueeze(0)
        out = self.model(tensor)
        idx_max = torch.argmax(out)
        result = idx_max.item() if out[idx_max] > self.threshold else -1
        if result == -1:
            print(f"Low confidence: {out[idx_max]}")
            print(f"Whole tensor: {out}")
        return result


    def save(file_path: str) -> None:
        raise NotImplementedError("Implement save method!")