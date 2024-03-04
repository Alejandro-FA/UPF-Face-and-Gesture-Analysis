import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import FaceRecognitionPipeline as frp
from PIL import Image
import os
import cv2
from imageio.v2 import imread

pipeline = frp.Pipeline(
    frp.FaceDetectorPreprocessor(output_channels=3),
    frp.MTCNNDetector(use_gpu=False, thresholds=[0.6, 0.7, 0.7]),
    # frp.MediaPipeDetector(model_asset_path="models/detector.tflite"),
    # frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3, color_transform=None),
    frp.FeatureExtractorPreprocessor(new_size=128, output_channels=3, color_transform=None),
    frp.DeepLearningExtractor(model_path="models/transfer_learning/superlight_v4_lab_norm/epoch-200.ckpt", num_classes=80, input_channels=3, use_gpu=False),
    detection_min_prob=0.9,
    classification_min_prob=0.4,
)

image = imread("data/TRAINING/image_A0007.jpg")
pipeline.visualize(image, "assets/pipeline_visualization")
