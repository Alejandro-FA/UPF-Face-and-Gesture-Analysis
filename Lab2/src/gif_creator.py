from PIL import Image
import os

# Directory containing your PNG images

for i in range(15):
    image_dir = f'assets/eigenvector_{i}'

    # Get a list of all image files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=lambda x: int(x.split('.')[0]))

    # Read the images using Pillow
    images = [Image.open(os.path.join(image_dir, file)) for file in image_files]

    # Specify the output file name
    output_file = os.path.join(image_dir, 'output.gif')

    # Save the images as a GIF

    images[0].save(output_file, save_all=True, append_images=images[0:], duration=100, loop=0)
