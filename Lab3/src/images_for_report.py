from PIL import Image
import os

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



# Usage:
base_path = os.path.join('assets')


image_files = [
    'distance_distance_plot_2.png',
    'distance_distance_plot_all.png',
]
result = stitch_images([os.path.join(base_path, x) for x in image_files])
result.save(os.path.join(RESULTS_PATH, "two_dist_dist_plot.png"), quality=1)