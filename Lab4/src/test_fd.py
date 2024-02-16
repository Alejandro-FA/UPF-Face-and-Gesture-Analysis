from __future__ import print_function
import numpy as np
from imageio.v2 import imread
from scipy.io import loadmat
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from FaceRecognitionPipeline import FaceDetector, MTCNNDetector, MediaPipeDetector
from tqdm import tqdm
import argparse


def CHALL_AGC_ComputeDetScores(DetectionSTR, AGC_Challenge1_STR, show_figures):
    #  Compute face detection score
    #
    #   INPUTS
    #     - DetectionSTR: A structure with the results of the automatic detection
    #     algorithm, with one element per input image containing field
    #     'det_faces'. This field contains as many 4-column rows as faces
    #     returned by the detector, each specifying a bounding box coordinates
    #     as [x1,y1,x2,y2], with x1 < x2 and y1 < y2.
    #
    #     - AGC_Challenge1_STR: The ground truth structure (e.g.AGC_Challenge1_TRAINING or AGC_Challenge1_TEST).
    #
    #     - show_figures: A flag to enable detailed displaying of the results for
    #     each input image. If set to zero it just conputes the scores, with no
    #     additional displaying.
    #
    #   OUTPUT
    #     - FD_score:     The final detection score obtained by the detector
    #     - scoresSTR:    Structure with additional detection information
    #   --------------------------------------------------------------------
    #   AGC Challenge
    #   Universitat Pompeu Fabra
    #
    feature_list = ['F1', 'Fmatrix', 'FNR']
    values = np.zeros((len(AGC_Challenge1_STR), 3), dtype='float')
    scoresSTR = pd.DataFrame(values, index=np.arange(len(AGC_Challenge1_STR)), columns=feature_list, dtype='object')
    
    for i in range(0, len(AGC_Challenge1_STR)):
        if show_figures:
            A = imread(AGC_Challenge1_STR['imageName'][i])
            fig, ax = plt.subplots()
            ax.imshow(A)
            for k1 in range(0, len(AGC_Challenge1_STR['faceBox'][i])):
                if len(AGC_Challenge1_STR['faceBox'][i]) != 0:
                    bbox = np.array(AGC_Challenge1_STR['faceBox'][i][k1], dtype=int)
                    fb = Rectangle((bbox[0], bbox[3]), bbox[2] - bbox[0], bbox[1] - bbox[3], linewidth=4, edgecolor='b',
                                   facecolor='none')
                    ax.add_patch(fb)
            for k2 in range(0, len(DetectionSTR[i])):
                if len(DetectionSTR[i]) != 0:
                    bbox = np.array(DetectionSTR[i][k2], dtype=int)
                    fb = Rectangle((bbox[0], bbox[3]), bbox[2] - bbox[0], bbox[1] - bbox[3], linewidth=4, edgecolor='g',
                                   facecolor='none')
                    ax.add_patch(fb)
        n_actualFaces = len(AGC_Challenge1_STR['faceBox'][i])
        n_detectedFaces = len(DetectionSTR[i])
        if not n_actualFaces:
            if n_detectedFaces:
                scoresSTR['F1'][i] = np.zeros(n_detectedFaces)
            else:
                scoresSTR['F1'][i] = np.array([1], dtype=float)
        else:
            if not n_detectedFaces:
                scoresSTR['F1'][i] = np.zeros(n_actualFaces)
            else:
                scoresSTR['Fmatrix'][i] = np.zeros((n_actualFaces, n_detectedFaces))
                for k1 in range(0, n_actualFaces):
                    f = np.array(AGC_Challenge1_STR['faceBox'][i][k1], dtype=int)
                    for k2 in range(0, n_detectedFaces):
                        g = np.array(DetectionSTR[i][k2], dtype=int)
                        # Intersection box
                        x1 = max(f[0], g[0])
                        y1 = max(f[1], g[1])
                        x2 = min(f[2], g[2])
                        y2 = min(f[3], g[3])
                        # Areas
                        int_Area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                        total_Area = (f[2] - f[0]) * (f[3] - f[1]) + (g[2] - g[0]) * (g[3] - g[1]) - int_Area
                        if n_detectedFaces == 1 and n_actualFaces == 1:
                            scoresSTR['Fmatrix'][i] = int_Area / total_Area
                        else:
                            scoresSTR['Fmatrix'][i][k1, k2] = int_Area / total_Area
                scoresSTR['F1'][i] = np.zeros((max(n_detectedFaces, n_actualFaces)))
                for k3 in range(0, min(n_actualFaces, n_detectedFaces)):
                    max_F = np.max(scoresSTR['Fmatrix'][i])
                    if n_detectedFaces == 1 and n_actualFaces == 1:
                        scoresSTR['F1'][i] = np.array([max_F], dtype=float)
                        scoresSTR['Fmatrix'][i] = 0
                        scoresSTR['Fmatrix'][i] = 0
                    else:
                        max_ind = np.unravel_index(np.argmax(scoresSTR['Fmatrix'][i], axis=None), scoresSTR['Fmatrix'][i].shape)
                        scoresSTR['F1'][i][k3] = max_F
                        scoresSTR['Fmatrix'][i][max_ind[0], :] = 0
                        scoresSTR['Fmatrix'][i][:, max_ind[1]] = 0

        false_negatives = n_actualFaces - np.count_nonzero(scoresSTR['F1'][i])
        scoresSTR['FNR'][i] = false_negatives / n_actualFaces if n_actualFaces else 0
        
        if show_figures:
            try:
                plt.title("%.2f" % scoresSTR['F1'][i])
            except:
                plt.title('%.2f, %.2f' % (scoresSTR['F1'][i][0], scoresSTR['F1'][i][1]))
                
            plt.show()
            plt.clf()
            plt.close()

    FD_score = np.mean(np.hstack(np.array(scoresSTR['F1'][:])))
    FNR_score = np.mean(np.hstack(np.array(scoresSTR['FNR'][:])))
    return FD_score, scoresSTR['F1'], FNR_score, scoresSTR['FNR']


def MyFaceDetectionFunction(A, model: FaceDetector):
    return [m.bounding_box.get_coords() for m in model(A)]


def save_scores(output_path: str, bounding_boxes, FD_score, f1_scores):
    # Save all F1 scores for analysis
    with open(output_path + "_scores.pkl", "wb") as file:
        pickle.dump(f1_scores, file)

    # Save bounding_boxes for analysis
    with open(output_path + "_bounding_boxes.pkl", "wb") as file:
        pickle.dump(bounding_boxes, file)

    with open(output_path + "_scores.txt", 'wt') as file:
        file.write(f'Overall F1-score: {FD_score}\n\n')
        for score, image_path in zip(f1_scores, AGC_Challenge1_TRAINING['imageName']):
            file.write(f'{image_path}: {score}\n')


def get_args():
    parser = argparse.ArgumentParser(description="AGC Challenge 1")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--results_path", required=False, help="Path prefix to the results file")
    parser.add_argument("--show_figures", required=False, help="Whether to show figures at the end or not", action='store_true', default=False)
    parser.add_argument("--disable_logs", required=False, help="Flag to disable debugging information", action='store_true', default=False)
    return parser.parse_args() 


def load_model(model_name: str) -> FaceDetector:
    valid_names = ["mtcnn", "mediapipe"]
    if model_name == valid_names[0]:
        return MTCNNDetector()
    elif model_name == valid_names[1]:
        return MediaPipeDetector("model/detector.tflite")
    else:
        raise RuntimeError(f"Invalid model name {model_name}. Valid names: {valid_names}")


if __name__ == '__main__':
    # Parse program arguments
    args = get_args()
    MODEL_NAME = args.model
    OUTPUT_FILE = args.results_path
    SHOW_FIGURES = args.show_figures
    if OUTPUT_FILE:
        print(f"Results will be stored in {OUTPUT_FILE}")

    # Basic script for Face Detection Challenge
    # --------------------------------------------------------------------
    # AGC Challenge
    # Universitat Pompeu Fabra

    # Load challenge Training data
    dir_challenge = "data/"
    AGC_Challenge1_TRAINING = loadmat(dir_challenge + "AGC_Challenge3_Training.mat")

    AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING['AGC_Challenge3_TRAINING'])
    AGC_Challenge1_TRAINING = [[row.flat[0] if row.size == 1 else row for row in line] for line in AGC_Challenge1_TRAINING]
    columns = ['id', 'imageName', 'faceBox']
    AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)

    # Provide the path to the input images, for example
    # 'C:/AGC_Challenge/images/'
    imgPath = dir_challenge + "TRAINING/"
    AGC_Challenge1_TRAINING['imageName'] = imgPath + AGC_Challenge1_TRAINING['imageName'].astype(str)
    # Initialize results structure
    DetectionSTR = []

    model = load_model(MODEL_NAME)

    # Initialize timer accumulator
    total_images = len(AGC_Challenge1_TRAINING)
    total_time = 0
    for idx, im in tqdm(enumerate(AGC_Challenge1_TRAINING['imageName']), total=total_images):
        A = imread(im)
        try:
            ti = time.time()
            # Timer on
            ###############################################################
            # Your face detection function goes here. It must accept a single
            # input parameter (the input image A) and it must return one or
            # more bounding boxes corresponding to the facial images found
            # in image A, specificed as [x1 y1 x2 y2]
            # Each bounding box that is detected will be indicated in a
            # separate row in det_faces

            det_faces = MyFaceDetectionFunction(A, model)

            tt = time.time() - ti
            total_time = total_time + tt
        except Exception as e:
            # print('Problematic image:', im) # FIXME: remove this for submission
            # raise e # FIXME: remove this for submission
            # If the face detection function fails, it will be assumed that no
            # face was detected for this input image
            det_faces = []

        DetectionSTR.append(det_faces)

    FD_score, f1_scores, FNR_score, fnr_scores = CHALL_AGC_ComputeDetScores(DetectionSTR, AGC_Challenge1_TRAINING, show_figures=SHOW_FIGURES)
    _, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('F1-score: %.2f, Total time: %2d m %.2f s' % (100 * FD_score, int(minutes), seconds))
    print('FNR-score: %.2f' % (100 * FNR_score))

    if OUTPUT_FILE:
        save_scores(OUTPUT_FILE, bounding_boxes=DetectionSTR, FD_score=FD_score, f1_scores=f1_scores)
