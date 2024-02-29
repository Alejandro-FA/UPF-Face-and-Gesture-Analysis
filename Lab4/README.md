# Lab 4. Face Recognition

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml &&
conda activate face_analysis &&
conda config --env --add channels conda-forge &&
conda config --env --add channels pytorch &&
conda config --env --add channels nvidia # Only needed if you have an Nvidia GPU
```

### pip

Alternatively, we also provide a `requirements.txt` file that can be used with `pip`. Please take into account that the project has been developed with `python 3.11`. We have not tested if the code works with other versions of `python`. To **replicate the development environment** simply run the following commands:

```bash
python3 -m venv .venv &&
source .venv/bin/activate &&
python -m pip install -r requirements.txt
```

## Execution instructions

With a terminal opened in the root folder of the lab, you only need to run the following command:

```bash
python src/CHALL_AGC_FRbasicScript.py 
```

This script will load on the CPU the DL model that we have trained and will start recognizing the identity of the people in each of the images stored in the `data/TRAINING` directory.

> NOTE: by default, the models will be loaded on the CPU. If you want to execute them in the GPU, you need to change the `load_model()` function for the following one:
>
> ```python
> def load_model() -> frp.Pipeline:
>    pipeline = frp.Pipeline(
>        frp.FaceDetectorPreprocessor(output_channels=3),
>        frp.MTCNNDetector(use_gpu=True, thresholds=[0.6, 0.7, 0.7]),
>        # frp.MediaPipeDetector(model_asset_path="model/detector.tflite"),
>        frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3),
>        frp.DeepLearningExtractor(model_path="model/transfer_learning/lab4_version/model_4-15.ckpt", num_classes=80, input_channels=3, use_gpu=True),
>        detection_min_prob=0.5, # Increasing this value to 0.9 improves the accuracy
>        classification_min_prob=0.4,
>    )
>    print(f"Loaded model with {pipeline.feature_extractor.num_parameters()} parameters")
>    return pipeline
> ```
> We strongly recommend <span style="color:red">NOT TO DO THIS</span> unless you have CUDA installed. We have not tested the code with other GPUs and we can not guarantee that it will run without errors.

## Face detector

We have used the MTCNN face detector.


## Feature extractor architecture

We have developed a light version of the LightCNN network, implemented by Wu, X. et al, which we have named SuperLightCNN. The main characteristics of the model concerning the requirements of the assignment are the following:
- Number of parameters: 957,808. All of them are trainable.
- Number of layers: 10
- Model files size: 3.83MB

More information can be found in `model/transfer_learning/lab4_version/model_4.txt`.

## Dataset

The model has been trained using a separate dataset. We have chosen to use the CelebA dataset. During the training process, the model has received 191261 corresponding to 10125 identities. It has been tested on 10125 images of the same dataset corresponding to 10125 identities (one testing image per identity).

Once the feature extractor has been trained to properly classify the 10K identities we have re-trained the last fully connected layer (the one responsible of the classification) with an extended version of the original 1200 training images. We have added 6527 images of the 80 identities that we have to recognize for the challenge. The original 1200 images are used just for testing the performance of the model once trained.

## Results

The F1 score obtained for the TRAINING dataset that we were provided is of 79.18%. It is important to remark that the model has not seen these images before, neither during the firs training process, neither during the transfer learning for the last fully connected layer. For this reason, we can interpret this result as an indication that we have managed to train a model that generalized fairly good. The following table shows the time that it takes to process the 1200 images on different CPUs.


| CPU                            | Time        |
| ------------------------------ | ----------- |
| 12th Gen Intel® Core™ i7-12700 | 0 m 52.05 s |
| 2,3 GHz Intel Core i9 8 cores  | 3 m 43.64 s |
| Ryzen 9 7900X                  | 0 m 37.14 s |
| Apple M1 Pro                   | 1 m 48.71 s |

## Different models tried

This is a summary of the performance of the new models with the VGG-Face2 dataset and the EXPANDED_v2 dataset

| MODEL NAME      | EPOCH LOWER LOSS | ACCURACY EPOCH LOWER LOSS | TL EPOCH LOWER LOSS | TL ACCURACY EPOCH LOWER LOSS | F1-score | Parameters |   TL Dataset   |
| --------------- | ---------------- | ------------------------- | ------------------- | ---------------------------- | -------- | ---------- | ---- |
| superlight_vgg2_expandedv2 | 13               | 79.2191435768262 %        | 50                  | 75.25%                       | 88.98    | 957808     | EXPANDED_v2     |
| superlight_vgg2_expanded | 13               | 79.2191435768262 %        | 50                | 88.75 %                | 89.82 | 957808 | EXPANDED     |
| superlight_lab_expandedv2 | 10               | 79.7338736173475 %        | 50 | 74.0 % | 86.01 | 957808 | EXPANDED_v2 |
| superlight_lab_expanded | 10 | 79.7338736173475 % | 50 | 86.875 % | 86.47 | 957808 | EXPANDED |
|                 |                  |                           |                     |                              |          |            |      |

### superlight_vgg2 and superlight_lab (both expanded and expanded_v2)

When used, be careful not to include LAB color transformation neither in the challenge nor in the transfer learning script.


Observations: the following is the code that should be placed in the `superlight_cnn.py` file for the model to work.

```python
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
    
```





## References
- MTCNN: https://pypi.org/project/facenet-pytorch/
- LightCNN: Wu, X., He, R., Sun, Z., & Tan, T. (2018). A light CNN for deep face representation with noisy labels. IEEE Transactions on Information Forensics and Security, 13(11), 2884-2896.
- LightCNN implementation: https://github.com/AlfredXiangWu/LightCNN
- CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- SqueezeNet: Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360.
- SqueezeNet implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
- Image scraper: https://github.com/ohyicong/Google-Image-Scraper