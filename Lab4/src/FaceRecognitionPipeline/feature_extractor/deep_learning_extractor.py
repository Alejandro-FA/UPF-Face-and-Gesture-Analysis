from .feature_extractor import FeatureExtractor
import os
import torch
from torchvision import transforms
import imageio.v2
from .superlight_cnn import superlight_network_9layers
import MyTorchWrapper as mtw
import matplotlib.pyplot as plt


class DeepLearningExtractor(FeatureExtractor):
    def __init__(self, model_path: str, num_classes=80, input_channels=3, use_gpu: bool = False) -> None:
        super().__init__()
        if not os.path.isfile(model_path):
            raise ValueError(f"Invalid file {model_path}")
        
        device = mtw.get_torch_device(use_gpu=use_gpu, debug=False)
        self.torch_transform = transforms.ToTensor()
        # self.model = network_9layers(num_classes=num_classes, input_channels=input_channels)
        self.model = superlight_network_9layers(num_classes=num_classes, input_channels=input_channels)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model_path = model_path


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

    
    def visualize(self, image: imageio.v2.Array, output_dir: str):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        
        image = transform(image).unsqueeze(0)
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook


        conv_layers = [layer_name for layer_name, _ in self.model.named_modules() if isinstance(_, torch.nn.Conv2d)]

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_forward_hook(get_activation(name))

        output = self.model(image)

        # Visualize the feature maps for each convolutional layer
        for layer_name, feature_map in activation.items():
            print(f"Creating feature maps for layer: {layer_name}")
            num_features = feature_map.size(1)
            num_cols = 8  # Number of feature maps to display per row
            num_rows = num_features // num_cols + 1

            plt.figure(figsize=(20, 20))
            plt.suptitle(f'Feature maps for layer: {layer_name}')
            for i in range(num_features):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(feature_map[0, i].cpu().detach().numpy(), cmap='jet')
                plt.axis('off')
            
            plt.savefig(f"{output_dir}/{layer_name}.png", dpi=500)

