from .feature_extractor import FeatureExtractor
import os
import torch
from torchvision import transforms
import imageio.v2
from .light_cnn import network_9layers


class DeepLearningExtractor(FeatureExtractor):
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