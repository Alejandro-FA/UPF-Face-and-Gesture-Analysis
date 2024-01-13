# LAB 1 - Face Detection Challenge

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda activate face_analysis
```

### pip

Alternatively, we also provide a `pip` `requirements.txt` file. Please take into account that the project has been developed with `python 3.11`. We have not tested if the code works with other versions of `python`. To **replicate the development environment** simply run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execution instructions FIXME: Improve this section

First run this command to generate the results (bounding boxes + f1 scores):

```bash
python src/CHALL_AGC_FDbasicScript.py results-2/test
```

Then run the following command to visualize the bounding boxes on top of the images. You can iterate over the images using the `A` and `D` keys of your keyboard.

```bash
python src/FD_results.py \
--ground_truth data/AGC_Challenge1_Training.mat \
--images data/TRAINING \
--generated_results results-2/test_bounding_boxes.pkl \
--scores results-2/test_scores.pkl
```

## Notes

No need to train any model. We can use an already existing Viola-Jones like algorithm.
We can use a pre-trained OpenCV model: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html


## TODO

- If a large face is not detected, try to look for an eye to see ensure that we do not have false positives.

- We could also use `alt2` method for detecting large faces, and take the largest face

- Search for profile faces as well.