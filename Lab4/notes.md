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

# Preprocessing models

- [MTCNN face detector](https://github.com/kpzhang93/MTCNN_face_detection_alignment): recommended in VGGFace2 paper.

# Datasets

- [DigiFace1M](https://github.com/microsoft/DigiFace1M?tab=readme-ov-file): 720K images with 10K identities (72 images per identity). For each identity, 4 different sets of accessories are sampled and 18 images are rendered for each set.
500K images with 100K identities (5 images per identity). For each identity, only one set of accessories is sampled. Problem: images are synthetic.
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 200K celebrity image with 10k identities. Faces can be downloaded in the wild, or aligned and cropped.
- [MillionCelebs](https://buptzyb.github.io/MillionCelebs): 22.8M images with 719K identities, downloadable with OneDrive. Faces are already cropped.

- [IMDB-Face](https://github.com/fwang91/IMDb-Face): 1.7M faces, 59K identities

- [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download)




