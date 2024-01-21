from PIL import Image
import os
import argparse
from tqdm import tqdm


def create_gif(path: str, output_path: str, eigenvec_num: int, delete_png=False):
        
    image_files = sorted(
        [f for f in os.listdir(path) if f.endswith('.png')],
        key=lambda x: int(x.split('.')[0])
    )

    
    images = [Image.open(os.path.join(path, file)) for file in image_files]

    
    output_file = os.path.join(f"{output_path}/", f'output_{eigenvec_num}.gif')

    images[0].save(output_file, save_all=True, append_images=images[0:], duration=100, loop=0)
    
    if delete_png:
        for file in image_files:
            os.remove(os.path.join(path, file))
    



parser = argparse.ArgumentParser(description="Creates a GIF for each of the modes of variation for the significant eigenvectors")
parser.add_argument("base_path", type=str, help="Base path where the files are stored")
parser.add_argument("num_components", type=int, help="Number of components for which the GIF have to be created")
parser.add_argument("--delete_png", action="store_true", help="Delete PNG files after creating the GIFs")
parser.add_argument("--landmarks", action="store_true", help="Create GIFs for landmarks")

args = parser.parse_args()

if not os.path.exists(args.base_path):
    print(f"Path {args.base_path} does not exist")
    exit(-1)

if args.landmarks:
    output_path = f"{args.base_path}/landmarks_gifs"
else:
    output_path = f"{args.base_path}/gifs"
    
if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in tqdm(range(args.num_components), total=args.num_components):
    if args.landmarks:
        curr_path = f"{args.base_path}/landmark_eigenvector_{i}"
    else:
        curr_path = f"{args.base_path}/eigenvector_{i}"
        
    if os.path.exists(curr_path):
        create_gif(curr_path, output_path, i, args.delete_png)
    else:
        print(f"Omitting {curr_path}. Path not found")