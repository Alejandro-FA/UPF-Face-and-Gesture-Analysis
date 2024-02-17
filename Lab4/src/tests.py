from imageio.v2 import imread
from pathlib import Path
import cv2
import FaceRecognitionPipeline as frp
import os
from facenet_pytorch import MTCNN
import MyTorchWrapper as mtw

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# path = "data/datasets/CelebA/Img/img_celeba/000001.jpg"

prep1 = frp.FaceDetectorPreprocessor(grayscale=False)
prep2 = frp.FeatureExtractorPreprocessor(new_size=128, grayscale=True)
# path = "data/TRAINING/image_A0017.jpg"
path = "data/TRAINING"
# path = "data/TRAINING/image_A0134.jpg"f
# path = "data/TRAINING/image_A0003.jpg"

mp_detector = frp.MediaPipeDetector("model/detector.tflite")
# mtcnn_detector = frp.MTCNNDetector()
device = mtw.get_torch_device(debug=True, use_gpu=True)
mtcnn_detector = MTCNN(image_size=128, post_process=True, keep_all=True, device=device)

buffer = []
batch_size = 128
faces = []
for image in sorted(os.listdir(path)):
    print(image)
    test_image = imread(f"{path}/{image}")
    buffer.append(prep1(test_image))

    if len(buffer) >= batch_size:
        faces.extend(mtcnn_detector(buffer), save_path=f"assets/{image}", return_prob=False)
        buffer = []
    # print(len(mt_cnn_res))
    # for res in mt_cnn_res:
    #     pass
        # tensor = prep2(test_image, res.bounding_box)
        # print(tensor.shape)


# print(type(mt_cnn_res))
# print(mt_cnn_res.shape)


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