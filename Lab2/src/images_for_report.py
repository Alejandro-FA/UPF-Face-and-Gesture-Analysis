from PIL import Image
import os
import numpy as np
import cv2
from eigenfaces.utils.image import ImagePreprocessor
from eigenfaces.utils.landmarks import Landmarks, LandmarksPreprocessor
from eigenfaces.utils.cfd_loader import CFDLoader
from eigenfaces.utils.visualizer import Visualizer

DATA_PATH = 'data'
RESULTS_PATH = 'assets'
DOWNSAMPLE_SIZE = (1222, 859)


def stitch_images(image_paths):
    images = [Image.open(x) for x in image_paths]

    max_height = max(img.height for img in images)

    images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in images]

    total_width = sum(img.width for img in images)
    new_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img


def find_landmark(landmarks: list[Landmarks], file_name: str) -> Landmarks:
    """
    Finds landmark coordinates corresponding to a given file name.

    Parameters:
        file_name (str): The file name for which to find landmark coordinates.

    Returns:
        list: A list of landmark coordinates.
    """
    
    for landmark in landmarks:
        if landmark.path.split(".")[0] == file_name.split(".")[0]:
            return landmark


# Paths
cfd_dir = os.path.join(DATA_PATH, 'CFD Version 3.0')
landmarks_dir = os.path.join(DATA_PATH, 'landmark_templates_01-29.22')

# Create preprocessors
# p = list(range(145, 158)) + [183, 184] + list(range(135, 144))
# landmarks_preprocessor = LandmarksPreprocessor(new_scale=DOWNSAMPLE_SIZE, unwanted_points=p)
# landmarks_preprocessor2 = LandmarksPreprocessor(new_scale=DOWNSAMPLE_SIZE)
# image_preprocessor = ImagePreprocessor(new_size=DOWNSAMPLE_SIZE)

# # Load data
# loader = CFDLoader(cfd_dir, landmarks_dir, image_preprocessor, landmarks_preprocessor)
# landmarks = loader.get_landmarks()
# images = loader.get_images()
# print(f'Loaded {len(images)} images and {len(landmarks)} landmarks')

# loader2 = CFDLoader(cfd_dir, landmarks_dir, image_preprocessor, landmarks_preprocessor2)
# landmarks2 = loader2.get_landmarks()
# print(f'Loaded {len(images)} images and {len(landmarks2)} landmarks')

# Visualize
# image_visualizer = Visualizer(images, landmarks2)
# image_visualizer.visualize(show_images=True, show_landmarks=True, show_landmarks_idx=True)

# show_images = True
# show_landmarks_idx = True
# idx = 525
# curr_image = images[idx]
# curr_landmarks = find_landmark(landmarks2, curr_image.path.split("/")[-1])

# l = Landmarks.from_vector(curr_landmarks.as_vector(), index_mapping=landmarks[0].index_mapping, joint_points=landmarks[0].joint_points, input_path=images[idx].path)

# if show_images:
#     image_coords = curr_image.as_matrix().copy()
# else:
#     image_coords = np.ones((DOWNSAMPLE_SIZE[1], DOWNSAMPLE_SIZE[0], 3), dtype=np.uint8) * 255
    
# for i, (x, y) in enumerate(curr_landmarks.as_matrix()):
#     image_coords = cv2.circle(image_coords, (x, y), 2, (0, 0, 0), thickness=-1)
#     if show_landmarks_idx:
#         image_coords = cv2.putText(image_coords, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)

# cv2.imshow(curr_image.path, image_coords)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Usage:
base_path = os.path.join('assets')
image_files = [
    'image_and_landmarks.png',
    'old_landmarks.png',
    'new_landmarks.png',
]
result = stitch_images([os.path.join(base_path, x) for x in image_files])
result.save(os.path.join(RESULTS_PATH, 'original_vs_preprocessed_landmarks.jpg'))