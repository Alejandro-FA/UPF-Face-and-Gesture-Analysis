# LAB 2 - Eigenfaces

## 1. Introduction

## 2. Theoretical background

### 2.1. Preprocessing

### 2.2. Eigenfaces algorithm

## 3. Dataset

We use the Chicago Faces Dataset. Images and landmarks from model `BF-209` have been removed because the dimensions (`2436 x 1712 x 3`) do not match the dimensions of the rest of the pictures (`2444 x 1718 x 3`).

## 3. Results

(include criterion to determine meaningfulness and representation of the 10 extracted bases in each feature space)

## 4. References
1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of cognitive neuroscience, 3(1), 71-86.

## NOTES

We have to argue if we use the pseudo-covariance matrix (that is an NxN matrix, where N is the number of samples) or the normal covariance matrix. For Eigenfaces we will use the pseudo, and for landmarks the normal one.

For eigenfaces, we need to normalize images (for example, we need to normalize the illumination, background conditions, rotations...). The ideal option is to find a dataset that it is already preprocessed.

Procrustes analysis for the preprocessing of the landmarks.

We can use built-in PCA methods for verification

Perhaps Quim likes a simple `if-else` condition inside PCA to determine whether to use the normal covariance matrix or the pseudo-covariance matrix (`N < image * width * channels`)

To visualize the principal components, ensure not to have any negative value and use the correct range.

Very important to explain modes of variation of the first principal directions.



## SUBMISSION INSTRUCTIONS

The literature review should be about 2-3 related papers and it must be around 10 lines long.

At most 5-6 pages, where 1 of them can be used for references and another one can be used for captions, images, tables...

At least we should use 1000 images. It will be relevant for the number of statistically significant principal components.

http://www.ifp.illinois.edu/~vuongle2/helen/

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

Dataset with 3700 images: https://github.com/StephenMilborrow/muct

Databases guide: https://libguides.princeton.edu/facedatabases

Chicago faces: https://www.chicagofaces.org/

Chicago faces landmarks: https://link.springer.com/article/10.3758/s13428-022-01830-7
