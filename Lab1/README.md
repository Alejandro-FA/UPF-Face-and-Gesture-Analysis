# LAB 1 - Face Detection Challenge

## Installation instructions

### conda

To **replicate the development environment** simply run the following commands (you can change the name of the environment from `face_analysis` to something else):

```bash
conda env create --name face_analysis --file environment.yml
conda activate face_analysis
```

### pip

Alternatively, we also provide a `requirements.txt` file that can be used with `pip`. Please take into account that the project has been developed with `python 3.11`. We have not tested if the code works with other versions of `python`. To **replicate the development environment** simply run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execution instructions

With a terminal opened in the root folder of the lab, you only need to run the following command:

```bash
python src/CHALL_AGC_FDbasicScript.py
```

By default, the model will save a log that indicates how many faces have been detected in each image and with which method. You can disable this behaviour with the `--disable_logs` flag. This log is intented to be used for debugging purposes, and it is important to note that not all detections are added to the final result. Additionaly, if you want to save the bounding boxes and the f1-scores of each image for later analysis, you can do so with the following command:

```bash
mkdir results
python src/CHALL_AGC_FDbasicScript.py --results_path results/test
```

Then you can run the following script to visualize the bounding boxes on top of the images. You can iterate over the images using the `A` and `D` keys of your keyboard.

```bash
python src/FD_results.py \
--ground_truth data/AGC_Challenge1_Training.mat \
--images data/TRAINING \
--generated_results results/test_bounding_boxes.pkl \
--scores results-2/test_scores.pkl
```

## Brief note on methodology

In this face detection challenge, the goal was to achieve a high average F1-score (minimum of `80.0`) in as little time as possible (a maximum of 15 minutes for 600 photos was allowed). The motivation of the project was to get familiarized with the Viola and Jones algorithm for detecting faces.

We were encouraged to use Haar cascade filters for the task, but we quickly realized that a single `CascadeClassifier` from `OpenCV` was not enough to achieve a high accuracy. In particular, we realized that we had a lot of **false positives**. In order to reduce the number of false positives, we decided to re-check candidate faces with other cascade classifiers, like `lbpcascade_frontalface_improved`, `haarcascade_eye_tree_eyeglasses` and `haarcascade_smile`, all from from OpenCV as well. So, in a sense, we have built a cascade of cascade classifiers, because further steps are only executed if a candidate face is found, and they only analyze a smaller region of interest.

Finally, we have also been able to improve the accuracy by rotating the images and re-analyzing them. In order to reduce the computation cost of this task, we only do this for images where no faces are found. Since this step produced more false positives than other steps, we have used a Deep Neural Netowrk (`mediapipe` from Google) for the verification step.

## Results

| F1-score  | Computation time (600 images)  |
|---|---|
| 87.47 | 1 m 30 s (Apple Macbook Pro M1) |


## Submission notes

No need to train any model. We can use an already existing Viola-Jones like algorithm.
We can use a pre-trained OpenCV model: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
