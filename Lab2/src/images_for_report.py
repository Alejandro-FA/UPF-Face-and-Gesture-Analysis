from PIL import Image
import os

RESULTS_PATH = 'assets'

def stitch_images(image_paths):
    # Open all images using PIL
    images = [Image.open(x) for x in image_paths]

    # Find the maximum height among all images
    max_height = max(img.height for img in images)

    # Resize images so they all have the same height
    images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in images]

    # Create a new image with a width equal to the sum of all image widths
    total_width = sum(img.width for img in images)
    new_img = Image.new('RGB', (total_width, max_height))

    # Paste images into the new image
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

# Usage:
base_path = os.path.join('data', 'CFD Version 3.0', 'Images', 'CFD')
image_files = [
    'AF-212/CFD-AF-212-097-N.jpg',
    'LM-252/CFD-LM-252-076-N.jpg',
    'WF-034/CFD-WF-034-006-N.jpg',
    'BM-010/CFD-BM-010-003-N.jpg'
]
result = stitch_images([os.path.join(base_path, x) for x in image_files])
result.save(os.path.join(RESULTS_PATH, 'stitched_image.jpg'))