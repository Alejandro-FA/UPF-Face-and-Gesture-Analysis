import numpy as np
# from imageio import imread
from imageio.v2 import imread
from scipy.io import loadmat
import random
import time
import itertools
from tqdm import tqdm
import pandas as pd
import FaceRecognitionPipeline as frp
import argparse
import os
import cv2
from torchvision import transforms


SUMMARY_FILE_NAME = "summary"
SUMMARY_PATH = "."
SUMMARY_FILE_EXTENSION = "txt"

def get_args():
    parser = argparse.ArgumentParser(description="Face recognition challenge")
    parser.add_argument("--summary", action=argparse.BooleanOptionalAction, default=False, help="After the execution of the challenge, generate a summary of the performance of the model")
    parser.add_argument("--detector_threshold", type=float, default=0.9, help="Threshold for the face detector")
    parser.add_argument("--classifier_threshold", type=float, default=0.4, help="Threshold for the face classifier")
    
    return parser.parse_args()


def create_summary_dict(ground_truth: list[int], detected_ids: list[int]):
    
    summary_dict = {}
    
    for real_id, detected_id in zip(ground_truth, detected_ids):
        if real_id not in summary_dict:
            summary_dict[real_id] = {"correct": 0, "total": 0, "fp": 0, "fp_ids": []}
        if detected_id not in summary_dict:
            summary_dict[detected_id] = {"correct": 0, "total": 0, "fp": 0, "fp_ids": []}
        
        summary_dict[real_id]["total"] += 1

        if detected_id == real_id:
            summary_dict[real_id]["correct"] += 1
        else:
            summary_dict[detected_id]["fp"] += 1
            if real_id not in summary_dict[detected_id]["fp_ids"]:
                summary_dict[detected_id]["fp_ids"].append(real_id)

    return summary_dict


def sort_categories(summary_dict):
    """
    Sorts the categories based on the correct classification
    """
    categories = [(id, info["correct"] / info["total"]) for id, info in summary_dict.items()]
    sorted_categories = sorted(categories, key=lambda x:x[1])
    
    return sorted_categories


def save_summary_dict(summary_dict, model_path: str, f1_score: float, time: str):
    summary_num = get_summary_num(SUMMARY_PATH)
    
    categories_correct_classifications = sort_categories(summary_dict)
    
    file = open(f"{SUMMARY_FILE_NAME}_{summary_num}.{SUMMARY_FILE_EXTENSION}", "w")
    file.write(f"Will create summary for the model\n\t{model_path}\n")
    file.write(f"F1-score: {f1_score}. Total time: {time}\n\n")
    
    cat_less_40 = []
    cat_40_60 = []
    cat_60_80 = []
    cat_more_80 = []
    
    # Classify categories acording to their tp rate
    for id, correct_rate in categories_correct_classifications:
        if correct_rate < 0.4:
            cat_less_40.append(id)
        elif correct_rate >= 0.4 and correct_rate < 0.6:
            cat_40_60.append(id)
        elif correct_rate >= 60 and correct_rate < 0.8:
            cat_60_80.append(id)
        else:
            cat_more_80.append(id)
    
    file.write(f"Categories with <40% of correct classifications ({len(cat_less_40)}):\n{cat_less_40}\n")
    file.write(f"Categories with 40%-60% of correct classifications ({len(cat_40_60)}):\n{cat_40_60}\n")
    file.write(f"Categories with 60%-80% of correct classifications ({len(cat_60_80)}):\n{cat_60_80}\n")
    file.write(f"Categories with >80% of correct classifications: ({len(cat_more_80)})\n{cat_more_80}\n\n")
    
    # TODO: if summary found useful, this could be written based on the number of correct classifications for each category
    for id, info in summary_dict.items():
        correct = info["correct"]
        total = info["total"]
        fp = info["fp"]
        fp_ids = info["fp_ids"]
        file.write(f"Category with ID {id}\n")
        file.write(f"\tClassified correctly {correct} out of {total} ({round(correct / total, 3) * 100}%)\n")
        file.write(f"\tFalse positives (number of people that have been classified as {id} without corresponding to it): {fp}\n")
        if fp > 0:
            file.write(f"\t\tIDs that have been classified as {id}: {fp_ids}\n\n")
        else:
            file.write("\n\n")
    
    file.close()

def get_summary_num(path: str):
    nums = []
    for file in os.listdir(path):
        if os.path.isfile(file) and file.startswith(SUMMARY_FILE_NAME):
            filename_no_ext = os.path.splitext(file)[0]
            nums.append(int(filename_no_ext[len(SUMMARY_FILE_NAME) + 1:]))
    
    if nums == []: return 1
    else: return max(nums) + 1

def CHALL_AGC_ComputeRecognScores(auto_ids, true_ids):
    #   Compute face recognition score
    #
    #   INPUTS
    #     - AutomSTR: The results of the automatic face
    #     recognition algorithm, stored as an integer
    #
    #     - AGC_Challenge_STR: The ground truth ids
    #
    #   OUTPUT
    #     - FR_score:     The final recognition score
    #
    #   --------------------------------------------------------------------
    #   AGC Challenge
    #   Universitat Pompeu Fabra
    #

    if len(auto_ids) != len(true_ids):
        assert ('Inputs must be of the same len')

    f_beta = 1
    res_list = list(filter(lambda x: true_ids[x] != -1, range(len(true_ids))))

    nTP = len([i for i in res_list if auto_ids[i] == true_ids[i]])

    res_list = list(filter(lambda x: auto_ids[x] != -1, range(len(auto_ids))))

    nFP = len([i for i in res_list if auto_ids[i] != true_ids[i]])

    res_list_auto_ids = list(filter(lambda x: auto_ids[x] == -1, range(len(auto_ids))))
    res_list_true_ids = list(filter(lambda x: true_ids[x] != -1, range(len(true_ids))))

    nFN = len(set(res_list_auto_ids).intersection(res_list_true_ids))

    FR_score = (1 + f_beta ** 2) * nTP / ((1 + f_beta ** 2) * nTP + f_beta ** 2 * nFN + nFP)

    return FR_score


def load_model(detector_threshold: float, classifier_threshold: float) -> frp.Pipeline:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4964, 0.5473, 0.5568], std=[0.1431, 0.0207, 0.0262]),
    ])

    pipeline = frp.Pipeline(
        frp.FaceDetectorPreprocessor(output_channels=3),
        frp.MTCNNDetector(use_gpu=False, thresholds=[0.6, 0.7, 0.7]),
        # frp.MediaPipeDetector(model_asset_path="models/detector.tflite"),
        frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3, color_transform=cv2.COLOR_RGB2LAB),
        frp.DeepLearningExtractor(model_path="models/transfer_learning/model_1/epoch-200.ckpt", num_classes=80, input_channels=3, use_gpu=False, torch_transform=transform),
        detection_min_prob=detector_threshold,
        classification_min_prob=classifier_threshold,
    )
    print(f"Loaded model with {pipeline.feature_extractor.num_parameters()} parameters")
    return pipeline

def my_face_recognition_function(A, my_FRmodel):
    return my_FRmodel(A)


# Basic script for Face Recognition Challenge
# --------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra
arguments = get_args()

summary = arguments.summary


# Load challenge Training data
dir_challenge3 = "data/"
AGC_Challenge3_TRAINING = loadmat(dir_challenge3 + "AGC_Challenge3_Training.mat")
AGC_Challenge3_TRAINING = np.squeeze(AGC_Challenge3_TRAINING['AGC_Challenge3_TRAINING'])

# Convert to dataframe (if needed)
AGC_Challenge3_TRAINING_df = [[row.flat[0] if row.size == 1 else row for row in line] for line in AGC_Challenge3_TRAINING]
columns = ['id', 'imageName', 'faceBox']
AGC_Challenge3_TRAINING_df = pd.DataFrame(AGC_Challenge3_TRAINING_df, columns=columns)


grouped = AGC_Challenge3_TRAINING_df.groupby(by=["id"])

# for id, data in grouped:
#     print(id[0])
#     for name in data["imageName"]:
#         print(f"\t{name}")


imageName = AGC_Challenge3_TRAINING['imageName']
imageName = list(itertools.chain.from_iterable(imageName))

ids = list(AGC_Challenge3_TRAINING['id'])
ids = np.concatenate(ids).ravel().tolist()

faceBox = AGC_Challenge3_TRAINING['faceBox']
faceBox = list(itertools.chain.from_iterable(faceBox))

imgPath = dir_challenge3 + "TRAINING/"

# Initialize results structure
AutoRecognSTR = []


# Load your FRModel
my_FRmodel = load_model(arguments.detector_threshold, arguments.classifier_threshold)

# Initialize timer accumulator
total_images = len(imageName)
total_time = 0
for idx, im in tqdm(enumerate(imageName), total=total_images):
    
    A = imread(imgPath + im)

    try:
        ti = time.time()
        # Timer on
        ###############################################################
        # Your face recognition function goes here.It must accept 2 input parameters:

        # 1. the input image A
        # 2. the recognition model

        # and must return a single integer number as output, which can be:

        # a) A number between 1 and 80 (representing one of the identities in the training set)
        # b) A "-1" indicating that none of the 80 users is present in the input image

        autom_id = my_face_recognition_function(A, my_FRmodel)
        # autom_id = -1

        tt = time.time() - ti
        total_time = total_time + tt
    except Exception as e:
        # print("Problematic image:", im) #FIXME: remove before submitting
        # raise e #FIXME: remove before submitting
        # If the face recognition function fails, it will be assumed that no user was detected for his input image
        autom_id = random.randint(-1, 80)

    AutoRecognSTR.append(autom_id)

FR_score = CHALL_AGC_ComputeRecognScores(AutoRecognSTR, ids)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
total_time = f"{int(minutes):2d} m {seconds: .2f} s"
FR_score = 100 * FR_score
print(f'F1-score: {FR_score:.2f}, Total time: {total_time}')


if summary:
    print(f"Summary number {get_summary_num(SUMMARY_PATH)} stored.")
    summary_dict = create_summary_dict(ids, AutoRecognSTR)
    save_summary_dict(summary_dict, my_FRmodel.feature_extractor.model_path, FR_score, total_time)