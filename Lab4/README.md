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

This script will load the DL model that we have trained on the CPU and will start recognizing the identity of the people in each of the images stored in the `data/TRAINING` directory.

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

Once the feature extractor has been trained to properly classify the 10K identities we have re-trained the last fully connected layer (the one responsible of the classification) with an extended version of the dataset (done by us) that we were provided with consisting of 6527 images of the 80 identities that we have to recognize for the challenge.

## Results

The F1 score obtained for the TRAINING dataset that we were provided is of 79.18%. It is important to remark that the model has not seen these images before, neither during the firs training process, neither during the transfer learning for the last fully connected layer. For this reason, we can interpret this result as an indication that we have managed to train a model that generalized fairly good. The following table shows the time that it takes to process the 1200 images on different CPUs.


| CPU                            | Time        |
| ------------------------------ | ----------- |
| 12th Gen Intel® Core™ i7-12700 | 0 m 52.05 s |
| 2,3 GHz Intel Core i9 8 cores  | 3 m 43.64 s |
| Ryzen 9 7900X                  | 0 m 37.14 s |
| Apple M1 Pro                   | 1 m 48.71 s |



## References
- MTCNN: https://pypi.org/project/facenet-pytorch/
- LightCNN: Wu, X., He, R., Sun, Z., & Tan, T. (2018). A light CNN for deep face representation with noisy labels. IEEE Transactions on Information Forensics and Security, 13(11), 2884-2896.
- LightCNN implementation: https://github.com/AlfredXiangWu/LightCNN
- CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- SqueezeNet: Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360.
- SqueezeNet implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py
- Image scraper: https://github.com/ohyicong/Google-Image-Scraper