import argparse
from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imageio.v2 import imwrite, imread

def get_args():
    parser = argparse.ArgumentParser(description="ID visualizer")
    parser.add_argument("--info_path", type=str,  help="Path to where the information of the images are stored.", required=True)
    parser.add_argument("--image_path", type=str, help="Path to the directory where the images are stored.", required=True)
    parser.add_argument("--id", type=int, help="ID of the person that wants to be visualized.", required=True)
    return parser.parse_args() 



def verify_args(id, info_path, image_path):
    if not (id == -1 or (id >= 1 and id <= 80)):
        print(f"ID has to be either -1 or [1-80]")
        return False
        
    try:
        loadmat(info_path)
    except:
        print(f"Invalid path to .mat file: {info_path}")
        return False
    
    if not os.path.isdir(image_path):
        print(f"Invalid image directory path: {image_path}")
        return False
    
    return True





def display_images_in_grid(image_paths: list[str], id: int) -> plt.figure:
    #TODO: change to opencv
    num_images = len(image_paths)
    grid_size = (int(math.sqrt(num_images)), math.ceil(num_images / int(math.sqrt(num_images))))
    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            image = mpimg.imread(image_paths[i])
            ax.imshow(image)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Images for person with id: {id}", fontsize=14, fontweight="bold")
    return fig



if __name__ == "__main__":
    args = get_args()
    
    info_path: str = args.info_path
    image_path: str = args.image_path
    id: int = args.id
    
    print(info_path)
    print(image_path)
    print(id)
    
    if not verify_args(id, info_path, image_path):
        exit(-1)
    
    if image_path.endswith("/"):
        image_path = image_path[:-1]
    
    data = loadmat(info_path)
    data = np.squeeze(data['AGC_Challenge3_TRAINING'])

    df_like = [[row.flat[0] if row.size == 1 else row for row in line] for line in data]
    columns = ['id', 'imageName', 'faceBox']
    df = pd.DataFrame(df_like, columns=columns)

    grouped = df.groupby(by=["id"])
    names = grouped.get_group(id)["imageName"].tolist()
    
    OUTPUT_DIR = "data/TRAINING_no_id"
    for image_name in names:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        image = imread(image_path + f"/{image_name}")
        imwrite(f"{OUTPUT_DIR}/{image_name}", image)
    # print(names)
    
    
    # Process images
    image_paths = [f"{image_path}/{name}" for name in names]
    print(f"{len(image_paths)} images")
    
    fig = display_images_in_grid(image_paths, id)
    
    plt.show()

    