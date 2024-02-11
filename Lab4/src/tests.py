from face_detector.mediapipe_detector import MediaPipeDetector
from face_detector.mtcnn_detector import MTCNNDetector
from imageio.v2 import imread
from pathlib import Path
import cv2

# path = "data/datasets/CelebA/Img/img_celeba/000001.jpg"
path = "data/TRAINING/image_A0135.jpg"

test_image = imread(path)
cv_image = cv2.imread(path)

mp_detector = MediaPipeDetector("model/detector.tflite")
mtcnn_detector = MTCNNDetector()

mp_res = mp_detector(test_image)
print("Mediapipe detected")
mt_cnn_res = mtcnn_detector(test_image)
print("MTCNN detected")


def plot_bboxes(image, results, color):
    for res in results:
        bbox = res.bounding_box
        image = cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
    return image

image = plot_bboxes(cv_image, mp_res, (0, 0, 0))
image = plot_bboxes(cv_image, mt_cnn_res, (255, 255, 255))

cv2.imshow("image with bboxes", image)
cv2.waitKey(0)