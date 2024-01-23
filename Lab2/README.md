# LAB 2 - Eigenfaces

## Execution instructions

With a terminal opened in the root folder of the lab, you only need to run the following command:

```bash
python src/main.py
```

## File structure

```
.
├── assets
├── data
├── docs
├── pickles
└── src
    └── eigenfaces
        ├── tests
        └── utils
```

- `assets`: Where the images generated during the execution of the program are stored.
- `data`: Contains all the dataset related files (both the images and their corresponding landmarks).
- `pickles`: Important folder where some precomputations are stored. Once executed the program for the first time, this folder will contain four files:
  - images_pca.pkl: the results of applying the PCA to the set of images.
  - landmarks_pca.pkl: the results of applying the PCA to the set of landmarks.
  - images.pkl: all the images information.
  - landmarks.pkl: all the landmarks information.
- `src`: the source code of the program.

## Some additional useful programs

### `pca_test.py`
Very important script that performs unit tests to ensure that the PCA algorithm works as expected. The tests can be executed as follows:
```bash
python -m unittest src/eigenfaces/tests/pca_test.py
```


### `giff_creator.py`
This script combines all the images generated for the modes of variation of each component and produces the resulting GIFs. After executing the main program, its usage is very simple. GIFs can be generated either through

```bash
python src/gif_creator.py assets 15
```

or

```bash
python src/gif_creator.py assets 15 --landmarks
```
It is important to ensure that the images for each component are stored under the directories `base_path/image_eigenvector_i` or `base_path/landmarks_eigenvector_i` where `base_path/` is the directory where these folders are stored and  `i` is the component number.

For more information, execute
```bash
python src/gif_creator.py -h
```
### `images_for_report.py`
This script concatenates images and has been used to generated some of the figures included in the report.

## Important notes
- if the file `images_scree_plot.png` is present under the `assets` folder, we highly recommend to set the `COMPUTE_IMAGE_SCREE_PLOT` variable of the main program (`main.py`) to a **FALSE** value, as generating this plot takes a lot of time.

- we use the Chicago Faces Dataset. Images and landmarks from model `BF-209` have been removed because the dimensions (`2436 x 1712 x 3`) do not match the dimensions of the rest of the pictures (`2444 x 1718 x 3`).

- file `CFD_WM-257-161-N.tem` has a corrupted byte. Change the corrupted byte in the last line by any integer number and add an extra line so the total lines of the file is 325.


## Dataset references

Chicago faces: https://www.chicagofaces.org/

Chicago faces landmarks: https://link.springer.com/article/10.3758/s13428-022-01830-7
