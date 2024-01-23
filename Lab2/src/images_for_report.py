from PIL import Image
import os

RESULTS_PATH = 'assets'

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
base_path = os.path.join('data', 'CFD Version 3.0', 'Images', 'CFD')
image_files = [
    'WM-026/CFD-WM-026-001-N.jpg',
    'CFD-WM-026-001-N.jpg',
    'WM-214/CFD-WM-214-026-N.jpg',
    'CFD-WM-214-026-N.jpg'
]
result = stitch_images([os.path.join(base_path, x) for x in image_files])
result.save(os.path.join(RESULTS_PATH, 'original_vs_preprocessed.jpg'))