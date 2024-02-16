from imageio.v2 import imread
from pathlib import Path
import cv2
import FaceRecognitionPipeline as frp
import os

# path = "data/datasets/CelebA/Img/img_celeba/000001.jpg"

prep1 = frp.FaceDetectorPreprocessor(grayscale=False)
prep2 = frp.FeatureExtractorPreprocessor(new_size=128, grayscale=True)
# path = "data/TRAINING/image_A0017.jpg"
path = "data/TRAINING"
# path = "data/TRAINING/image_A0134.jpg"
# path = "data/TRAINING/image_A0003.jpg"

mp_detector = frp.MediaPipeDetector("model/detector.tflite")
mtcnn_detector = frp.MTCNNDetector()

for image in os.listdir(path):
    print(image)
    test_image = imread(f"{path}/{image}")
    mt_cnn_res = mtcnn_detector(prep1(test_image))
    for res in mt_cnn_res:
        tensor = prep2(test_image, res.bounding_box)
        # print(tensor.shape)




# mp_res = mp_detector(test_image)
# print("Mediapipe detected")

# def plot_bboxes(image, results, color):
#     for res in results:
#         bbox = res.bounding_box
#         image = cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
#     return image

# image = plot_bboxes(cv_image, mp_res, (0, 0, 0))
# image = plot_bboxes(cv_image, mt_cnn_res, (255, 255, 255))

# cv2.imshow("image with bboxes", image)
# cv2.waitKey(0)