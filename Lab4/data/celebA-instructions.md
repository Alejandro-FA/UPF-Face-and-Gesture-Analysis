Once you have downloaded the zip files from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, execute the following commands:

> NOTE: The 7z tool does not come installed by default in some operating systems. You might need to install it with your package manager.

## Join dataset

```bash
unzip 'CelebA-*.zip'
cd CelebA/Img
```

## Unzip Align&Cropped Images in JPG format

```bash
unzip 'img_align_celeba.zip'
```

## Unzip Align&Cropped Images in PNG format

```bash
cd 
7z x 'img_align_celeba_png.7z'
```

## Unzip In-The-Wild Images

```bash
7z x 'img_celeba.7z'
```
