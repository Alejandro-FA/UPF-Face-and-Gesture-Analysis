from __future__ import print_function
import numpy as np
from imageio.v2 import imread
from scipy.io import loadmat
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2 as cv
import sys
import pickle


if len(sys.argv) != 2:
    print("You must specify the name of the output file where the detected faces will be stored")
    print("Be aware that only one name is allowed")
    exit(-1)
else:
    output_file_name = sys.argv[1]

print(f"Results will be stored in {output_file_name}")

def CHALL_AGC_ComputeRates(DetectionSTR, AGC_Challenge1_STR):
    pass


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
    feature_list = ['F1', 'Fmatrix']
    values = np.zeros((len(AGC_Challenge1_STR), 2), dtype='float')
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
                        scoresSTR['F1'][i][max_ind[1]] = max_F
                        scoresSTR['Fmatrix'][i][max_ind[0], :] = 0
                        scoresSTR['Fmatrix'][i][:, max_ind[1]] = 0
        if show_figures:
            try:
                plt.title("%.2f" % scoresSTR['F1'][i])
            except:
                plt.title('%.2f, %.2f' % (scoresSTR['F1'][i][0], scoresSTR['F1'][i][1]))
                
            plt.show()
            plt.clf()
            plt.close()

    # Save all F1 scores for analysis
    with open(output_file_name + "_scores.pkl", "wb") as score_file:
        pickle.dump(scoresSTR["F1"], score_file)

    FD_score = np.mean(np.hstack(np.array(scoresSTR['F1'][:])))
    return FD_score



class Model:
    def __init__(self, haar_face_model: str, lbp_face_model: str, eyes_model: str) -> None:
        self.face_haarcascade = cv.CascadeClassifier()
        self.face_lbpcascade = cv.CascadeClassifier()
        self.eyes_cascade = cv.CascadeClassifier()
        
        try:
            self.eyes_cascade.load(cv.samples.findFile(eyes_model))
            self.face_haarcascade.load(cv.samples.findFile(haar_face_model))
            self.face_lbpcascade.load(cv.samples.findFile(lbp_face_model))
        except Exception:
            print('--(!)Error loading opencv file')
            exit(0)
        

    def detect_faces(self, image) -> list[tuple[int, int, int, int]]:
        frame_gray = self.preprocess(image)

        #-- Detect faces
        faces = self.face_lbpcascade.detectMultiScale(frame_gray)
        # eyes = self.eyes_cascade.detectMultiScale(frame_gray)
        detected_faces = []
        # print(len(faces))
        
        
        for (x,y,w,h) in faces:
            margin = 0.75
            y2_roi = int(y - margin * h)
            x2_roi = int(x - margin * w)
            faceROI = frame_gray[
                y2_roi : y + int((1 + margin)*h),
                x2_roi : x + int((1 + margin)*w)
            ]
            
            #-- In each face, detect eyes
            # eyes = self.eyes_cascade.detectMultiScale(faceROI)
            large_faces = self.face_haarcascade.detectMultiScale(faceROI)
            # If a face is detected with the accurate model, find a more suitable bounding box
            # print(len(large_faces))
            if len(large_faces) > 0:
                # detected_faces.append(self.__get_largest_faces(large_faces, 10))
                detected_large_faces = []
                for (x2,y2,w2,h2) in large_faces:
                    detected_large_faces.append(self.__get_box(x2 + x2_roi, y2 + y2_roi , w2, h2, image, 0))

                largest_face = self.__get_largest_faces(detected_large_faces, 1)[0]
                detected_faces.append(largest_face)
            else:
                detected_faces.append(self.__get_box(x,y,w,h,image, 0.3))
            # detected_faces.append(self.__get_box(x,y,w,h,image, 0.25))
            

            # if len(eyes) > 0:
            #     detected_faces.append((x, y, x + w, y + h))

            # for (x2, y2, w2, h2) in eyes:
            #     detected_faces.append((x+x2, y+y2, x+x2 + w2, y+y2 + h2))

        # return self.__get_largest_faces(detected_faces, 2)
        return detected_faces


    def preprocess(self, image):
        try:
            frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except:
            frame_gray = image
        frame_gray = cv.equalizeHist(frame_gray)
        return frame_gray


    def __get_largest_faces(self, faces: list[tuple[int, int, int, int]], n: int) -> list[tuple[int, int, int, int]]:
        largest_faces = sorted(
            faces,
            key= lambda x: (x[2] - x[0]) * (x[3] - x[1]),
            reverse=True
        )[0:n]
        
        return largest_faces
    

    def __get_box(self, x, y, w, h, image, margin: float) -> tuple[int, int, int, int]:
        img_h = image.shape[0]
        img_w = image.shape[1]
        return (
            self.__clamp(x - (0 + margin) * w, 0, img_w),
            self.__clamp(y - (0 + margin) * h, 0, img_h),
            self.__clamp(x + (1 + margin) * w, 0, img_w),
            self.__clamp(y + (1 + margin) * h, 0, img_h)
        )

    def __clamp(self, x: float, min_threshold, max_threshold) -> int:
        x = int(x)
        x = min(x, max_threshold)
        x = max(x, min_threshold)
        return x
    

def MyFaceDetectionFunction(A, model: Model):
    return model.detect_faces(A)


# Basic script for Face Detection Challenge
# --------------------------------------------------------------------
# AGC Challenge
# Universitat Pompeu Fabra

# Load challenge Training data
dir_challenge = "AGC_Challenge1_Materials/"
AGC_Challenge1_TRAINING = loadmat(dir_challenge + "AGC_Challenge1_Training.mat")

AGC_Challenge1_TRAINING = np.squeeze(AGC_Challenge1_TRAINING['AGC_Challenge1_TRAINING'])
AGC_Challenge1_TRAINING = [[row.flat[0] if row.size == 1 else row for row in line] for line in AGC_Challenge1_TRAINING]
columns = ['id', 'imageName', 'faceBox']
AGC_Challenge1_TRAINING = pd.DataFrame(AGC_Challenge1_TRAINING, columns=columns)


imgPath = dir_challenge + "TRAINING/"
AGC_Challenge1_TRAINING['imageName'] = imgPath + AGC_Challenge1_TRAINING['imageName'].astype(str)
# Initialize results structure
DetectionSTR = []

total_images = len(AGC_Challenge1_TRAINING)
info_every = 100

haar_face = "opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
lbp_face = "opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
eyes_path = "opencv/data/haarcascades_cuda/haarcascade_eye.xml"
model = Model(haar_face, lbp_face, eyes_path)

# Initialize timer accumulator
total_time = 0
for idx, im in enumerate(AGC_Challenge1_TRAINING['imageName']):
    if idx % info_every == 0:
        _, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"[{idx} / {total_images}] images processed. Elapsed time: {int(minutes)} m {seconds:.2f} s")
    
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
        print(im)
        det_faces = []
        raise e
        # If the face detection function fails, it will be assumed that no
        # face was detected for this input image

    DetectionSTR.append(det_faces)



# CHALL_AGC_ComputeRates(DetectionSTR, AGC_Challenge1_TRAINING)
FD_score = CHALL_AGC_ComputeDetScores(DetectionSTR, AGC_Challenge1_TRAINING, show_figures=False)
_, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print('F1-score: %.2f, Total time: %2d m %.2f s' % (100 * FD_score, int(minutes), seconds))

with open(output_file_name + "_bounding_boxes.pkl", "wb") as output:
    pickle.dump(DetectionSTR, output)
