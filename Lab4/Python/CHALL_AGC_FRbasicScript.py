import os
import numpy as np
# from imageio import imread
from imageio.v2 import imread
from scipy.io import loadmat
import random
import time
import itertools


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
        assert ('Inputs must be of the same len');

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


def my_face_recognition_function(A, my_FRmodel):
    # Function to implement
    print()


# Basic script for Face Recognition Challenge
# --------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra

# Load challenge Training data
dir_challenge3 = " "
AGC_Challenge3_TRAINING = loadmat(dir_challenge3 + "AGC_Challenge3_Training.mat")
AGC_Challenge3_TRAINING = np.squeeze(AGC_Challenge3_TRAINING['AGC_Challenge3_TRAINING'])

imageName = AGC_Challenge3_TRAINING['imageName']
imageName = list(itertools.chain.from_iterable(imageName))

ids = list(AGC_Challenge3_TRAINING['id'])
ids = np.concatenate(ids).ravel().tolist()

faceBox = AGC_Challenge3_TRAINING['faceBox']
faceBox = list(itertools.chain.from_iterable(faceBox))

imgPath = " "

# Initialize results structure
AutoRecognSTR = []

# Initialize timer accumulator
total_time = 0

# Load your FRModel
my_FRmodel = " "

for idx, im in enumerate(imageName):

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

        tt = time.time() - ti
        total_time = total_time + tt
    except:
        # If the face recognition function fails, it will be assumed that no user was detected for his input image
        autom_id = random.randint(-1, 80)

    AutoRecognSTR.append(autom_id)

FR_score = CHALL_AGC_ComputeRecognScores(AutoRecognSTR, ids)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print('F1-score: %.2f, Total time: %2d m %.2f s' % (100 * FR_score, int(minutes), seconds))
