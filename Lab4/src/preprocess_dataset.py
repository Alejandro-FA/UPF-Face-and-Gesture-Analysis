from imageio.v2 import imread, imwrite
import FaceRecognitionPipeline as frp
import os
import torch
from tqdm import tqdm


def get_ids(ids_path: str) -> dict[str, int]:
    """
    Loads the information from the ids file.
    A line has the following format:
        XXXXXX.jpg <id>
    where XXXXXX represents the image number, and <id> represents the id of the person in that image.
    """
    ids = {}
    try:
        file = open(ids_path, "r").read().strip()
    except:
        raise FileNotFoundError(f"Filt {ids_path} could not be found")

    for line in file.splitlines():
        splited_line = line.split(" ")
        img_num = splited_line[0].split(".")[0]
        id = int(splited_line[1])
        ids[img_num] = id
    
    return ids
    

def save_image(tensor: torch.Tensor, file_name: str, ids: dict[str, int], test_ids_count: dict[int, int]):
    THRESHOLD = 3
    TENSORS_TRAIN_PATH = "data/datasets/CelebA/Img/img_align_celeba_train"
    TENSORS_TEST_PATH = "data/datasets/CelebA/Img/img_align_celeba_test"

    if os.path.exists(TENSORS_TRAIN_PATH) == False:
        os.makedirs(TENSORS_TRAIN_PATH)

    if os.path.exists(TENSORS_TEST_PATH) == False:
        os.makedirs(TENSORS_TEST_PATH)
    
    image_name = file_name.split(".")[0]
    id = ids[image_name]
    
    if id not in test_ids_count:
        test_ids_count[id] = 1

    img_count = test_ids_count[id]
    if img_count > THRESHOLD:
        # Train image
        imwrite(f"{TENSORS_TRAIN_PATH}/{file_name}", tensor)
        # torch.save(tensor, f"{TENSORS_TRAIN_PATH}/{image_name}.pt")
    else:
        # Test image
        imwrite(f"{TENSORS_TEST_PATH}/{file_name}", tensor)
        # torch.save(tensor, f"{TENSORS_TEST_PATH}/{image_name}.pt")




if __name__ == '__main__':
    IMAGES_PATH = "data/datasets/CelebA/Img/img_align_celeba"
    ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA.txt"
    ids = get_ids(ANNOTATIONS_PATH)

    prep1 = frp.FaceDetectorPreprocessor(grayscale=False)
    prep2 = frp.FeatureExtractorPreprocessor(new_size=128, grayscale=False)

    mp_detector = frp.MediaPipeDetector("model/detector.tflite")
    mtcnn_detector = frp.MTCNNDetector()
    test_ids_count = {}

    max = None
    file_list = sorted(os.listdir(IMAGES_PATH))
    for count, image_path in tqdm(enumerate(file_list), total=len(file_list)):

        image = imread(f"{IMAGES_PATH}/{image_path}")
        mt_cnn_res = mtcnn_detector(prep1(image))
        
        for res in mt_cnn_res:
            tensor = prep2(image, res.bounding_box)
            save_image(tensor, image_path, ids, test_ids_count)

        if max is not None and count >= max:
            break



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