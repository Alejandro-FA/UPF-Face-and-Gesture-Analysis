from .feature_extractor import FeatureExtractor
import os
import torch
from torchvision import transforms
import imageio.v2
from .light_cnn import network_9layers
from .superlight_cnn import superlight_network_9layers
import MyTorchWrapper as mtw


class DeepLearningExtractor(FeatureExtractor):
    def __init__(self, model_path: str, num_classes=80, input_channels=3) -> None:
        super().__init__()
        if not os.path.isfile(model_path):
            raise ValueError(f"Invalid file {model_path}")
        
        self.torch_transform = transforms.ToTensor()
        # self.model = network_9layers(num_classes=num_classes, input_channels=input_channels)
        self.model = superlight_network_9layers(num_classes=num_classes, input_channels=input_channels)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    

    def __call__(self, image: imageio.v2.Array) -> tuple[int, float]:
        # Transform image to tensor
        tensor: torch.Tensor = self.torch_transform(image).unsqueeze(0)
        
        # Run inference on the model and get the probabilities
        output = self.model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get the class with the highest probability
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_class_prob = probabilities[0][predicted_class].item()
        return predicted_class + 1, predicted_class_prob # We add 1 to the class to match the expected output of the pipeline (1-indexed classes)


    def save(file_path: str) -> None:
        raise NotImplementedError("Implement save method!")
    
    
    def num_parameters(self):
        return mtw.get_model_params(self.model)