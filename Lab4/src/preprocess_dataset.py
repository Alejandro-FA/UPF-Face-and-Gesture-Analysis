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
    

def save_image(tensor: torch.Tensor, file_name: str, ids: dict[str, int], test_ids_count: dict[int, int], train_ids_count: dict[int, int]):
    THRESHOLD = 3
    TENSORS_TRAIN_PATH = "data/datasets/CelebA/Img/img_align_celeba_train"
    TENSORS_TEST_PATH = "data/datasets/CelebA/Img/img_align_celeba_test"

    if os.path.exists(TENSORS_TRAIN_PATH) == False:
        os.makedirs(TENSORS_TRAIN_PATH)

    if os.path.exists(TENSORS_TEST_PATH) == False:
        os.makedirs(TENSORS_TEST_PATH)
    
    image_name = file_name.split(".")[0]
    id = ids[image_name]
    
    img_count = test_ids_count[id]
    if img_count > THRESHOLD:
        # Train image
        train_ids_count[id] += 1
        torch.save(tensor, f"{TENSORS_TRAIN_PATH}/{image_name}.pt")
    else:
        # Test image
        test_ids_count[id] += 1
        torch.save(tensor, f"{TENSORS_TEST_PATH}/{image_name}.pt")


def print_stats(test_ids_count: dict[int, int], train_ids_count: dict[int, int]):
    LOG_PATH = "data/datasets/CelebA/log.txt"

    with open(LOG_PATH, "w") as file:
        unique_test_ids = [id for id, count in test_ids_count.items() if count > 0]
        unique_train_ids = [id for id, count in train_ids_count.items() if count > 0]
        file.write(f"Unique test ids: {len(unique_test_ids)}\n")
        file.write(f"Unique train ids: {len(unique_train_ids)}\n")

        for id, count in test_ids_count.items():
            file.write(f"ID: {id}, test count: {count}, train count: {train_ids_count[id]}\n")

        


if __name__ == '__main__':
    IMAGES_PATH = "data/datasets/CelebA/Img/img_align_celeba"
    ANNOTATIONS_PATH = "data/datasets/CelebA/Anno/identity_CelebA.txt"
    ids = get_ids(ANNOTATIONS_PATH)

    prep1 = frp.FaceDetectorPreprocessor(grayscale=False)
    prep2 = frp.FeatureExtractorPreprocessor(new_size=128, grayscale=False)

    mp_detector = frp.MediaPipeDetector("model/detector.tflite")
    mtcnn_detector = frp.MTCNNDetector()

    test_ids_count = {}
    train_ids_count = {}
    for id in ids.values():
        test_ids_count[id] = 0
        train_ids_count[id] = 0

    max = None
    file_list = sorted(os.listdir(IMAGES_PATH))
    for count, image_path in tqdm(enumerate(file_list), total=len(file_list)):
        if max is not None and count >= max:
            break

        image = imread(f"{IMAGES_PATH}/{image_path}")
        mt_cnn_res = mtcnn_detector(prep1(image))
        
        for res in mt_cnn_res:
            tensor = prep2(image, res.bounding_box)
            save_image(tensor, image_path, ids, test_ids_count, train_ids_count)

        if count % 1000 == 0:
            print_stats(test_ids_count, train_ids_count)
