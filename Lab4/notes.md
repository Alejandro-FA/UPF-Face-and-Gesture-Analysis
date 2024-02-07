# Notes

Avoid overfitting. Can be useful to plot train and test loss evolution.

Submodels: parallel branches (maximum of 2).

Deep Learning recommended specifically for the feature extractor step


Baseline:
- Viola-Jones Face Detection
- HOG Features
- SVM Classifier / Random Forest

Transfer learning will probably not pass the layer-parameter limitation.




# Ideas

- Deep Learning limits only apply to the feature extraction step.

- QUIM recommendation: use a flexible face detection (better to have low false negatives). The face recognition step will handle the face of cats as well in principle. If we use the face detection from the first lab, take into consideration that it had low false positives (but relatively high false negative).

- Cross validation

- Multiclass classification vs Binary classification (not recommended by Quim).

- Use detector from Lab 1 vs use an end-to-end system

- Leave alignment as a last resource (difficult and not very worth it)

- Data augmentation

- Explore transformers

- Think about how to handle grayscale images. Try to keep all 3 rgb channels

- Light-weight networks
  - Light CNN
  - MobilFace
  - SqueezeNet
  - ShuffleNet
  - Xception


- Create an id downloader. Keep each id in a separate folder