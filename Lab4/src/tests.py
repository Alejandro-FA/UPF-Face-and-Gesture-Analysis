from face_detector.mediapipe_detector import MediaPipeDetector
from face_detector.mtcnn_detector import MTCNNDetector
from imageio.v2 import imread
from pathlib import Path


test_image = imread(Path("data/datasets/CelebA/Img/img_celeba/000001.jpg"))

mp_detector = MediaPipeDetector(Path("model/detector.tflite"))
mtcnn_detector = MTCNNDetector()

mp_res = mp_detector(test_image)
print("Mediapipe detected")
mt_cnn_res = mtcnn_detector(test_image)
print("MTCNN detected")

print(mp_res)
print(mt_cnn_res)